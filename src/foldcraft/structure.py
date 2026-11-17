"""The Structure > Model > Chain > Residue > Atom hierarchy.

Chains keep residues in file order, addressable by :class:`ResidueId`
(sequence number plus insertion code); every child keeps a back-reference to
its parent, so an atom found by a neighbour search can report its chain.
"""

from __future__ import annotations

import copy as _copy
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import NamedTuple

from . import geometry as geo
from .constants import BACKBONE_ATOMS, THREE_TO_ONE, WATER_NAMES, one_letter
from .errors import StructureError
from .geometry import Mat3, Vec3

ResidueKey = int | tuple[int, str]


class ResidueId(NamedTuple):
    """Author residue identifier: sequence number and insertion code."""

    seq: int
    icode: str = ""

    def __str__(self) -> str:
        return f"{self.seq}{self.icode}"


@dataclass(eq=False)
class Atom:
    """An atomic site; ``coord`` in Angstrom, ``b_factor`` in A^2, ``altloc`` empty if absent."""

    name: str
    element: str
    coord: Vec3
    serial: int = 0
    occupancy: float = 1.0
    b_factor: float = 0.0
    altloc: str = ""
    charge: int = 0
    parent: Residue | None = field(default=None, repr=False)

    @property
    def x(self) -> float:
        return self.coord[0]

    @property
    def y(self) -> float:
        return self.coord[1]

    @property
    def z(self) -> float:
        return self.coord[2]

    @property
    def is_hydrogen(self) -> bool:
        return self.element in ("H", "D")

    def distance_to(self, other: Atom) -> float:
        return geo.distance(self.coord, other.coord)

    def label(self) -> str:
        """``chain:RESseq:atom`` label for reports."""
        res = self.parent
        chain = res.parent.id if res is not None and res.parent is not None else "?"
        return f"{chain}:{res.name}{res.id}:{self.name}" if res is not None else self.name


class Residue:
    """A residue: amino acid, nucleotide, ligand or solvent molecule."""

    def __init__(self, name: str, seq: int, icode: str = "", hetero: bool = False) -> None:
        self.name = name.upper()
        self.seq = int(seq)
        self.icode = icode.strip()
        self.hetero = hetero
        self.atoms: dict[str, Atom] = {}
        self.parent: Chain | None = None

    @property
    def id(self) -> ResidueId:
        return ResidueId(self.seq, self.icode)

    def add_atom(self, atom: Atom) -> Atom:
        """Attach *atom*; duplicate names raise :class:`StructureError`."""
        if atom.name in self.atoms:
            raise StructureError(f"duplicate atom {atom.name!r} in residue {self.name} {self.id}")
        atom.parent = self
        self.atoms[atom.name] = atom
        return atom

    def remove_atom(self, name: str) -> Atom:
        atom = self[name]
        del self.atoms[name]
        atom.parent = None
        return atom

    def get(self, name: str) -> Atom | None:
        return self.atoms.get(name)

    def __getitem__(self, name: str) -> Atom:
        try:
            return self.atoms[name]
        except KeyError:
            raise StructureError(f"no atom {name!r} in residue {self.name} {self.id}") from None

    def __contains__(self, name: object) -> bool:
        return name in self.atoms

    def __iter__(self) -> Iterator[Atom]:
        return iter(self.atoms.values())

    def __len__(self) -> int:
        return len(self.atoms)

    @property
    def is_water(self) -> bool:
        return self.name in WATER_NAMES

    @property
    def is_amino_acid(self) -> bool:
        """True for residues with an N-CA-C backbone or a standard polymer name."""
        if all(n in self.atoms for n in ("N", "CA", "C")):
            return True
        return not self.hetero and self.name in THREE_TO_ONE

    @property
    def one_letter(self) -> str:
        return one_letter(self.name)

    def has_backbone(self) -> bool:
        """Whether all four peptide backbone heavy atoms (N, CA, C, O) are present."""
        return all(n in self.atoms for n in BACKBONE_ATOMS)


class Chain:
    """An ordered collection of residues sharing a chain identifier."""

    def __init__(self, chain_id: str) -> None:
        self.id = chain_id
        self.residues: list[Residue] = []
        self.index: dict[ResidueId, Residue] = {}
        self.parent: Model | None = None

    @staticmethod
    def _key(key: ResidueKey) -> ResidueId:
        if isinstance(key, int):
            return ResidueId(key, "")
        if isinstance(key, tuple) and len(key) == 2:
            return ResidueId(int(key[0]), str(key[1]).strip())
        raise TypeError(f"invalid residue key {key!r}")

    def add_residue(self, residue: Residue) -> Residue:
        if residue.id in self.index:
            raise StructureError(f"duplicate residue {residue.id} in chain {self.id!r}")
        residue.parent = self
        self.residues.append(residue)
        self.index[residue.id] = residue
        return residue

    def remove_residue(self, key: ResidueKey) -> Residue:
        res = self[key]
        self.residues.remove(res)
        del self.index[res.id]
        res.parent = None
        return res

    def __getitem__(self, key: ResidueKey) -> Residue:
        rid = self._key(key)
        try:
            return self.index[rid]
        except KeyError:
            raise StructureError(f"no residue {rid} in chain {self.id!r}") from None

    def __contains__(self, key: object) -> bool:
        try:
            return self._key(key) in self.index  # type: ignore[arg-type]
        except TypeError:
            return False

    def __iter__(self) -> Iterator[Residue]:
        return iter(self.residues)

    def __len__(self) -> int:
        return len(self.residues)

    def atoms(self) -> Iterator[Atom]:
        for res in self.residues:
            yield from res

    def polymer(self) -> list[Residue]:
        """Amino-acid residues in chain order (ligands and water excluded)."""
        return [r for r in self.residues if r.is_amino_acid and not r.is_water]

    def sequence(self) -> str:
        """One-letter sequence of :meth:`polymer` residues."""
        return "".join(r.one_letter for r in self.polymer())


class Model:
    """A single coordinate model (one conformer of an ensemble)."""

    def __init__(self, serial: int = 1) -> None:
        self.serial = serial
        self.chains: dict[str, Chain] = {}
        self.parent: Structure | None = None

    def add_chain(self, chain: Chain) -> Chain:
        if chain.id in self.chains:
            raise StructureError(f"duplicate chain {chain.id!r} in model {self.serial}")
        chain.parent = self
        self.chains[chain.id] = chain
        return chain

    def get_or_create_chain(self, chain_id: str) -> Chain:
        return self.chains[chain_id] if chain_id in self.chains else self.add_chain(Chain(chain_id))

    def __getitem__(self, chain_id: str) -> Chain:
        try:
            return self.chains[chain_id]
        except KeyError:
            raise StructureError(f"no chain {chain_id!r} in model {self.serial}") from None

    def __iter__(self) -> Iterator[Chain]:
        return iter(self.chains.values())

    def __len__(self) -> int:
        return len(self.chains)

    def residues(self) -> Iterator[Residue]:
        for chain in self.chains.values():
            yield from chain

    def atoms(self) -> Iterator[Atom]:
        for chain in self.chains.values():
            yield from chain.atoms()


class Structure:
    """Top-level container: an identifier, models in file order, and header metadata."""

    def __init__(self, structure_id: str = "") -> None:
        self.id = structure_id
        self.models: list[Model] = []
        self.metadata: dict[str, object] = {}

    def add_model(self, model: Model) -> Model:
        if any(m.serial == model.serial for m in self.models):
            raise StructureError(f"duplicate model serial {model.serial}")
        model.parent = self
        self.models.append(model)
        return model

    @property
    def model(self) -> Model:
        """The first model."""
        if not self.models:
            raise StructureError("structure contains no models")
        return self.models[0]

    def __getitem__(self, index: int) -> Model:
        return self.models[index]

    def __iter__(self) -> Iterator[Model]:
        return iter(self.models)

    def __len__(self) -> int:
        return len(self.models)

    def residues(self) -> Iterator[Residue]:
        return self.model.residues()

    def atoms(self) -> Iterator[Atom]:
        """Atoms of the first model."""
        return self.model.atoms()

    def all_atoms(self) -> Iterator[Atom]:
        """Atoms of every model, in file order."""
        for m in self.models:
            yield from m.atoms()

    def transform(self, rotation: Mat3 = geo.IDENTITY, translation: Vec3 = geo.ORIGIN) -> None:
        """Apply ``x' = R x + t`` to every atom of every model, in place."""
        for atom in self.all_atoms():
            atom.coord = geo.add(geo.mat_vec(rotation, atom.coord), translation)

    def copy(self) -> Structure:
        """Deep copy with all parent links rebuilt."""
        return _copy.deepcopy(self)

    def summary(self) -> dict[str, int]:
        """Counts of models, and of chains, residues and atoms in the first model."""
        if not self.models:
            return {"models": 0, "chains": 0, "residues": 0, "atoms": 0}
        m = self.model
        return {
            "models": len(self.models),
            "chains": len(m.chains),
            "residues": sum(len(c) for c in m),
            "atoms": sum(1 for _ in m.atoms()),
        }


def ca_atoms(container: Structure | Model | Chain | Iterable[Residue]) -> list[Atom]:
    """C-alpha atoms of the amino-acid residues of *container*, in order."""
    if isinstance(container, (Structure, Model)):
        residues: Iterable[Residue] = container.residues()
    else:
        residues = container
    return [r.atoms["CA"] for r in residues if r.is_amino_acid and "CA" in r.atoms]
