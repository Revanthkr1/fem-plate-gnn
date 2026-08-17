# Project Flow — FEM Plate-with-Hole GNN Surrogate

Running log of decisions, results, and gotchas as this project is built. Kept
up to date phase by phase (see the approved plan for the original design);
this file is the "what actually happened" companion to that plan, and is
meant to double as raw material for the eventual README narrative.

## Context

Second, separate portfolio project alongside the AirfRANS CFD-GNN surrogate.
Same overall idea (MeshGraphNets-style GNN, geometry-in/field-out) applied to
FEM structural simulation instead of CFD — the toy version of what Neural
Concept sells commercially, and specifically covers the FEM side (Abaqus)
rather than CFD.

## Problem definition (decided with the user before building)

- **Geometry**: 2D rectangular plate (100mm x 200mm) with a randomized
  circular hole (radius + position).
- **Material**: elastoplastic (isotropic hardening) — no geometric
  nonlinearity (NLGEOM off).
- **Prediction target**: single-shot final converged state at one load level
  (not a load-history/sequence).
- **Output fields**: displacement (u_x, u_y) + von Mises stress, per node.
- **Node features** (planned): `[x, y, signed distance to hole boundary,
  load magnitude]` — mirrors AirfRANS's `[x, y, wall_distance, inlet_vx,
  inlet_vy]` pattern (position + geometry descriptor + broadcast per-case
  loading condition).

## Phase 0 — Environment check

- Abaqus **2023**, installed under `EstProducts` (the full commercial/
  research product line — not the free Student Edition, which caps models at
  1000 nodes). No node-count ceiling to worry about.
- The CAE scripting kernel runs on the bundled **Python 2.7** (not 3) —
  `caeModules`/`abaqus`/`odbAccess` scripts must be written in Python 2
  syntax.
- License server is `akash3.cc.iitk.ac.in:29000` (IIT Kanpur campus FlexNet
  server), reachable only over the campus network or the Fortinet VPN.
  **Any `abaqus cae`/job submission requires the VPN adapter to actually be
  connected** (it was seen disconnecting/showing "Disabled" mid-session more
  than once — worth checking first if a run inexplicably fails on a license
  error).

## Phase 1 — Single elastic case, verified against Kirsch

**Setup**: 100x200mm plate, 5mm-radius hole centered at (50,100), CPS4R plane
stress elements, global seed 2mm with a finer 0.5mm seed on the hole edge.
Bottom edge `u2=0` + bottom-left corner `u1=0` (rigid-body pin), top edge
`u2=0.05mm` applied displacement. Material: E=210000 MPa, nu=0.3 (elastic
only at this phase).

**Result**: FE peak von Mises = **142.76 MPa**; nominal remote stress (from
summed reaction force on the fixed edge / cross-section area) = **51.88
MPa** → FE stress concentration factor = **2.75** vs. the analytical Kirsch
solution of **3.0** (small hole in a plate under uniaxial tension) — **8.3%
error, within the 10% pass tolerance**. Confirms the geometry/material/BC/
mesh setup is fundamentally correct before adding any complexity.

Files: `src/abaqus_gen.py`, `src/check_kirsch.py`.

## Phase 2 — Add plasticity

Added an isotropic hardening table `(250 MPa, 0.0), (300 MPa, 0.01), (400
MPa, 0.05)` (yield stress vs. plastic strain) to the same material, and
raised the top displacement to 0.15mm — enough that the *elastic-equivalent*
peak stress (~430 MPa, scaling linearly from phase 1's result) would sit well
past the 250 MPa yield point.

**Result**: actual max von Mises capped at **260.14 MPa** (vs. ~430 MPa if it
had stayed purely elastic) — confirms real stress redistribution from
yielding. Max equivalent plastic strain (PEEQ) = **0.002**, localized to 61
of 8922 integration points — plastic yielding concentrated right at the hole
edge, exactly as physically expected, not spread through the whole plate.

File: `src/abaqus_gen.py` (updated in place), `src/check_plastic.py`.

## Phase 3 — Parametrized batch generation (in progress)

`src/generate_dataset.py` loops randomized `(hole_r, hole_x, hole_y, load)`
draws through a **single** CAE session (spawning `abaqus cae` per-case would
cost ~15-20s of pure startup overhead each time) — hole radius in [3,15]mm,
hole center kept >=12mm from every plate edge, load (top displacement) in
[0.05, 0.20]mm. Each case is built, meshed, solved, then immediately
extracted from its `.odb` into a compact `data/raw/case_NNN.json`
(node coordinates, element connectivity, per-node displacement, per-node
averaged von Mises), after which the heavy Abaqus job artifacts
(`.odb/.sta/.msg/.inp/...`) are deleted — mirrors the AirfRANS project's
cache-then-discard convention.

**Status**: a 2-case validation run now succeeds end-to-end. Case 0 (hole_r
= 13.13mm): 5706 nodes, 5606 elements (5464 quads + 142 triangular
transition elements from the quad-dominated free mesher around the circular
hole — expected, not a bug), all displacement/von-Mises values populated,
max von Mises 239.76 MPa. Scaling up to a real smoke-test batch (20-50
cases) next.

### Debugging notes (worth keeping — non-obvious Abaqus scripting gotchas hit along the way)

1. `abaqus cae noGUI=...` does **not** forward Python `print()` output to the
   invoking shell — progress/errors must be written to an explicit log file
   instead (see the `log()` helper in `generate_dataset.py`).
2. `mdb.models` rename/delete via `changeKey` has stale-reference pitfalls —
   fetching the model object *before* renaming it silently breaks later calls
   on that reference (`KeyError` on the old name). Worked around by giving
   each case its own uniquely-named model instead of reusing/deleting one
   shared name.
3. `job.status`, read off the object returned by `mdb.Job(...)`, can come
   back `None` even after `waitForCompletion()` even though the job actually
   completed — re-fetch via `mdb.jobs[job_name].status` instead.
4. The odb stores instance names **uppercased** regardless of the case used
   when creating them in CAE (created `'Plate-1'`, odb has `'PLATE-1'`).
5. Abaqus's own numeric types (node coordinates, field values) are not
   natively JSON-serializable — every extracted leaf value needs an explicit
   `float()`/`int()` cast.
6. Relative output paths are resolved against the *Abaqus process's* working
   directory at submission time, not the script's own file location — easy
   to get the `../..` depth wrong when the run directory nests differently
   than expected (hit this once: `../raw` vs. the correct `../../raw`).

## Repo layout

See the approved plan (`configs/`, `src/`, `notebooks/`, `data/` — same shape
as the AirfRANS project).

## Phase 4 — Mesh-to-graph, model, training pipeline (done)

`src/graph.py::build_graph(case)` ports the AirfRANS bidirectional-edge +
relative-position-edge-feature pattern exactly: Abaqus's mesh (CPS4R quads +
a few CPS3 triangles) is loaded into a PyVista `UnstructuredGrid`, and the
*same* `extract_all_edges()` call used for the CFD project's gmsh mesh gives
unique, deduplicated element edges. Node features are `[x, y,
signed_distance_to_hole, load_magnitude]` (4), targets are `[u_x, u_y,
von_mises]` (3) — mirrors AirfRANS's `[x, y, wall_distance, inlet_vx,
inlet_vy]` → 4-field pattern.

Validated directly against the extraction script's own output for case 0:
5706 nodes, 11312 unique undirected edges (5606 elements), max von Mises
239.76 MPa matching exactly.

`normalization.py`, the three-tier `dataset.py` (raw / PyG / cached-tensor —
minus AirfRANS's subsampling tier, unnecessary here since these meshes are
already small), `model.py` (MeshGraphNet, ported byte-identical except
`node_in_dim=4`/`out_dim=3`), `metrics.py` (`FIELD_NAMES = ["u_x", "u_y",
"von_mises"]`), and `train.py` (Lightning module — wall-distance-weighted
loss and surface-MAE metric dropped, no FEM equivalent; plain MSE training
loss) are all ported. `configs/base.yaml` is populated from day one (unlike
the AirfRANS repo, where this never happened) and is load-bearing —
`train.py`'s `__main__` entry point actually reads model/training
hyperparameters from it.

## Phase 6 — Local overfit test (done, gate passed)

Preprocessed all 10 generated cases to cached tensors, computed normalization
stats (`n_nodes=65377` across 10 cases; sane node/target mean-std). Trained
on 3 cases for 300 epochs (CPU, `latent_dim=32, hidden_dim=64,
n_message_passing=4` — `MeshGraphNet`'s own un-scaled-up defaults) with
train and val set to the *same* 3 cases (deliberate overfit check).

**Result**: per-field relative L2 on those 3 cases — `u_x`: 1.8-4.3%, `u_y`:
1.1-1.4%, `von_mises`: 1.3-2.9%. Confirms the full pipeline (Abaqus
generation → extraction → mesh-to-graph → normalization → GNN → training
loop) is correct end-to-end, per CLAUDE.md's "must overfit before tuning"
gate.

## Phase 7 — Scale dataset generation to 200 cases (done)

Scaling from the 10-case smoke test to the full 200 ran straight into a real
infrastructure constraint that had nothing to do with the code: the IIT
Kanpur campus Abaqus **`standard`** (solver) license pool is shared and
fluctuates hard with unrelated campus load — observed everywhere from
**80/80 tokens in use** (fully saturated, jobs queue indefinitely with no
ETA) down to **5/80** (wide open, cases complete in seconds) over the course
of a single day, unrelated to anything this project was doing. One `standard`
job was seen queuing for 8+ hours during a saturated period before being
cleaned up.

Response: switched from "check in every 15-30 min" to just queuing the
**entire** remaining range (cases 10-199) in one unattended background run —
Abaqus's own per-job license queue handles waiting for a token per case
automatically, and `generate_dataset.py`'s per-case try/except means one
slow/failed case never blocks the rest. Also added a `job_completed`
file-existence check (see `main()`'s `if os.path.exists(out_path)`) so a run
that gets interrupted (a session restart silently killed the background
process at least twice during this phase, leaving a stale `.lck`/partial job
behind — cleaned up by hand each time) can just be relaunched and pick up
where it left off rather than regenerating already-cached cases.

**Result**: all 190 remaining cases succeeded once run during a genuinely
open license window (5/80 standard tokens in use) — **0 failures/skips
across the full 190**, landing at **200/200 total cases** in `data/raw/`.

**Practical lesson for next time**: if a run stalls, check
`abaqus licensing lmstat -a` for the `standard` feature's usage count before
assuming anything is broken — campus load, not code, was the cause in every
stall observed here. Off-peak hours (evening, in this case) cleared the
contention within minutes of relaunching.

## Remaining phases

8. Move real training to Colab (200 full-resolution cases is well past what
   the local 4GB GPU should take on), mirroring the AirfRANS project's
   `colab_setup.ipynb`/`kaggle_setup.ipynb` pattern. Recompute normalization
   stats over the full 200 (currently `norm_stats.npz` only reflects the
   original 10).
9. Evaluate with per-field relative L2 on a held-out split (u_x, u_y,
   von_mises reported separately) plus the Kirsch-style physical sanity
   check applied per-case.
10. "Copilot" demo (geometry/parameters in, instant prediction out via the
    trained GNN — mirroring how Neural Concept's actual product works, not a
    natural-language agent) — comes after training and evaluation, not
    before.
11. README narrative, and — if desired — a PhysicsNeMo v2 port, mirroring
    the AirfRANS project's stated ambition.
