"""Format-independent assembly of a :class:`~foldcraft.structure.Structure`.

Both readers feed atom records to :class:`StructureBuilder`, which owns the
shared policies: alternate-location selection, repeated atom names, and
micro-heterogeneity (one position modelled as two residue types, of which the
first is kept). Non-fatal anomalies are collected in ``warnings``.
"""

from __future__ import annotations

from ..errors import StructureError
from ..structure import Atom, Model, Residue, ResidueId, Structure

AltlocPolicy = str  # "first", "occupancy", or a single altloc character


class StructureBuilder:
    """Assemble a structure from a stream of atom records.

    *altloc* is ``"first"`` (first conformer seen), ``"occupancy"`` (highest
    occupancy) or a single indicator such as ``"B"`` (that conformer only).
    With *strict*, duplicate atoms raise :class:`StructureError`; otherwise
    they are dropped and recorded in :attr:`warnings`.
    """

    def __init__(self, structure_id: str = "", altloc: AltlocPolicy = "first", strict: bool = True) -> None:
        if altloc not in ("first", "occupancy") and len(altloc) != 1:
            raise ValueError(f"invalid altloc policy {altloc!r}")
        self.structure = Structure(structure_id)
        self.altloc = altloc
        self.strict = strict
        self.warnings: list[str] = []
        self._model: Model | None = None
        self._skipped: set[tuple[int, str, ResidueId, str]] = set()

    def begin_model(self, serial: int) -> Model:
        """Make the model with *serial* current, creating it if necessary."""
        model = next((m for m in self.structure if m.serial == serial), None)
        self._model = model if model is not None else self.structure.add_model(Model(serial))
        return self._model

    def add_atom(self, atom: Atom, chain_id: str, resname: str, seq: int, icode: str = "",
                 hetero: bool = False, model_serial: int | None = None) -> Atom | None:
        """Insert *atom*, creating parents as needed; returns ``None`` if the atom was discarded."""
        if model_serial is not None or self._model is None:
            self.begin_model(model_serial if model_serial is not None else 1)
        model = self._model
        assert model is not None
        chain = model.get_or_create_chain(chain_id)
        rid, resname = ResidueId(seq, icode.strip()), resname.upper()
        key = (model.serial, chain_id, rid, resname)
        if key in self._skipped:
            return None
        residue = chain.index.get(rid)
        if residue is None:
            residue = chain.add_residue(Residue(resname, seq, icode, hetero))
        if residue.name != resname:
            if not atom.altloc or self.altloc in ("first", "occupancy") or atom.altloc == self.altloc:
                self.warnings.append(f"residue {chain_id}:{rid} modelled as both {residue.name} and {resname}; "
                                     f"keeping {residue.name}")
                self._skipped.add(key)
            return None
        existing = residue.get(atom.name)
        if not atom.altloc:
            if existing is not None:
                msg = f"duplicate atom {atom.name!r} in {residue.name} {chain_id}:{rid}"
                if self.strict:
                    raise StructureError(msg)
                self.warnings.append(msg + "; ignored")
                return None
        elif self.altloc == "first" or existing is None and self.altloc == "occupancy":
            if existing is not None:
                return None
        elif self.altloc == "occupancy":
            if atom.occupancy <= existing.occupancy:  # type: ignore[union-attr]
                return None
            residue.remove_atom(atom.name)
        elif atom.altloc != self.altloc:
            return None
        elif existing is not None:
            residue.remove_atom(atom.name)
        return residue.add_atom(atom)

    def finish(self) -> Structure:
        """Return the structure with empty residues and chains removed."""
        for model in self.structure:
            for chain_id, chain in list(model.chains.items()):
                for res in [r for r in chain if not r.atoms]:
                    chain.remove_residue(res.id)
                if not chain.residues:
                    del model.chains[chain_id]
        return self.structure
