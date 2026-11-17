# foldcraft

**foldcraft** is a pure-Python toolkit for reading, analysing and scoring
macromolecular structures. It has no runtime dependencies beyond the Python
standard library (3.10+), which makes it easy to vendor into structure
pipelines, containers and restricted compute environments.

## Features

| Area | Capabilities |
| --- | --- |
| **I/O** | PDB reader/writer (fixed columns, hybrid-36 serials and residue numbers, alternate locations, insertion codes, multi-model files, element inference, formal charges, gzip input); mmCIF reader/writer (CIF 1.1 tokenizer with quoted strings, semicolon text fields, `loop_` tables, null markers, `auth_*` / `label_*` identifiers) |
| **Hierarchy** | `Structure` > `Model` > `Chain` > `Residue` > `Atom`, with parent links, insertion-code-aware residue lookup, deep copies and rigid-body transforms |
| **Geometry** | Distances, angles, dihedrals, NeRF atom placement, cell-list neighbour search, Jacobi eigen-decomposition |
| **Topology** | Covalent bond graph with peptide, phosphodiester and disulfide linkage rules; chain-break detection |
| **Contacts** | Distance matrices, CA / CB / heavy-atom contact maps that double as residue graphs, sequence-separation ranges, relative contact order, precision/recall against a reference |
| **Secondary structure** | DSSP-style assignment (Kabsch & Sander H-bond energy; H, G, I, E, B, T, S), backbone phi/psi/omega |
| **Superposition** | Quaternion-based Kabsch superposition (weighted, reflection-free) and RMSD |
| **Model quality** | GDT_TS, GDT_HA, TM-score, steric clash detection with MolProbity-style clash score |
| **Building** | Ideal backbones from torsion angles and idealised parallel / antiparallel two-strand sheets |
| **CLI** | `foldcraft info | ss | contacts | clashes | compare | convert` |

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
```

## Quick start

```python
import foldcraft as fc

s = fc.load("1abc.cif.gz")            # format inferred from extension or content
print(s.summary())                    # {'models': 1, 'chains': 2, 'residues': 312, 'atoms': 2410}
chain = s.model["A"]
print(chain.sequence())

ss = fc.assign_dssp(chain)
print(ss.string)                      # e.g. "--HHHHHHHHHHTT--EEEEE--..."

cmap = fc.ContactMap.from_structure(s, cutoff=8.0, mode="CB", min_separation=6)
print(len(cmap.by_range("long")), cmap.relative_contact_order())

report = fc.find_clashes(s)
print(report.clash_score, report.worst.describe() if report.worst else "no clashes")
```

### Comparing a model with a reference

```python
model, ref = fc.load("model.pdb"), fc.load("reference.cif")
pairs = fc.pair_residues(model, ref, method="sequence")   # or "numbering", "order"
mob, tgt = fc.paired_coordinates(pairs, ("CA",))

sup = fc.kabsch(mob, tgt)
print(f"RMSD   {sup.rmsd:.3f} A over {sup.n_points} CA atoms")
print(f"GDT_TS {fc.gdt_ts(mob, tgt):.3f}")
print(f"TM     {fc.tm_score(mob, tgt).score:.3f}")
```

### Parsing policies

```python
fc.read_pdb("entry.pdb", altloc="occupancy")   # keep the highest-occupancy conformer
fc.read_pdb("entry.pdb", altloc="B")           # keep only conformer B
fc.read_pdb("entry.pdb", strict=False)         # skip malformed records, see metadata["warnings"]
fc.read_mmcif("entry.cif", use_author_fields=False)  # label_* chain and residue numbering
```

Strict mode (the default) raises `PDBParseError` / `MMCIFParseError` with the
offending line number and text.

## Command line

```text
$ foldcraft info 1abc.pdb
$ foldcraft ss 1abc.pdb --three-state
$ foldcraft contacts 1abc.cif --cutoff 8 --mode CB --min-sep 6 --summary
$ foldcraft clashes model.pdb --tolerance 0.4
$ foldcraft compare model.pdb reference.cif --pairing sequence --json
$ foldcraft convert 1abc.pdb 1abc.cif
```

## Methods and references

* **Secondary structure** follows W. Kabsch & C. Sander, *Biopolymers* 22,
  2577 (1983). Hydrogen bonds use the electrostatic energy
  `E = 0.084 * 332 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN)` kcal/mol with a
  -0.5 kcal/mol threshold.
* **Superposition** uses the quaternion formulation of B. K. P. Horn,
  *J. Opt. Soc. Am. A* 4, 629 (1987), and E. A. Coutsias, C. Seok & K. A.
  Dill, *J. Comput. Chem.* 25, 1849 (2004).
* **GDT** follows A. Zemla, *Nucleic Acids Res.* 31, 3370 (2003); **TM-score**
  follows Y. Zhang & J. Skolnick, *Proteins* 57, 702 (2004). Both use a
  fragment-seeded iterative maximal-subset search; values agree closely with,
  but are not guaranteed identical to, the reference programs.
* **Ideal geometry** uses R. A. Engh & R. Huber, *Acta Cryst.* A47, 392
  (1991); atom placement uses the NeRF method of J. Parsons et al.,
  *J. Comput. Chem.* 26, 1063 (2005).
* **Radii**: covalent radii from B. Cordero et al., *Dalton Trans.* 2832
  (2008); van der Waals radii from A. Bondi, *J. Phys. Chem.* 68, 441 (1964).

## Scope and limitations

* The mmCIF reader covers the categories needed for coordinates and common
  header items; it is not a full dictionary-validating CIF parser.
* Pure-Python geometry favours portability over speed; the cell-list
  neighbour search keeps contact, clash and H-bond detection near-linear, but
  very large assemblies are better served by vectorised libraries.
* Secondary-structure assignment does not model polyproline helices or
  chirality-based assignments introduced in later DSSP versions.

## License

MIT; see `LICENSE`.
