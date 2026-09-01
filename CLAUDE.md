# FEM Plate-with-Hole GNN Surrogate

Portfolio project: a MeshGraphNets-style GNN surrogate trained on Abaqus FEM
simulations of a 2D plate with a randomized circular hole, predicting
displacement + von Mises stress fields directly from mesh geometry and load.
Sibling project to the AirfRANS CFD-GNN surrogate (same overall idea, FEM/
structural instead of CFD) — see `PROJECT_FLOW.md` for the running build log
and the plan this was built from.

## Problem definition

- Geometry: 100mm x 200mm plate, randomized circular hole (radius + position).
- Material: elastoplastic (isotropic hardening), no geometric nonlinearity.
- Prediction target: single-shot final converged state at one load level
  (not a load-history/sequence).
- Output fields: displacement (u_x, u_y) + von Mises stress, per node.

## Stack

- Abaqus 2023 (`abaqus cae noGUI=...` for the CAE scripting kernel, Python
  2.7 — not 3) for data generation: parametrized geometry/mesh/material/BCs,
  job submission, `.odb` extraction via `odbAccess`.
- PyTorch Geometric (model, ported from the AirfRANS `MeshGraphNet`), PyVista
  (mesh/graph construction), Lightning (training loop).
- License server is IIT Kanpur campus FlexNet (`akash3.cc.iitk.ac.in`) —
  requires the Fortinet VPN to be connected for any Abaqus run.

## Layout

- `src/` — all real logic (Abaqus generation/extraction scripts, graph
  construction, model, training, metrics). Same convention as AirfRANS:
  nothing important should live only in a notebook.
- `notebooks/` — exploration and plotting only.
- `configs/` — hyperparameters and run configs (populated from the start,
  unlike the AirfRANS repo where this never happened).
- `data/` — `raw/` (per-case JSON extracted from Abaqus) and `norm_stats.npz`
  are committed to git: unlike AirfRANS's dataset (~15GB, downloaded from an
  external package), this dataset is small (~200MB for 200 cases) and has no
  external source — it only exists wherever it was Abaqus-generated, so
  committing it is what makes `git clone` alone enough to reproduce training
  elsewhere (e.g. Colab). `runs/` (transient Abaqus job working directories)
  and `cache/` (preprocessed tensors, regeneratable via
  `src/recompute_stats.py`) are gitignored, along with checkpoints -- with
  one deliberate, narrow exception: `data/model_release.ckpt` (the phase
  11c2 checkpoint) is committed, since the demo (`src/app.py`) needs a
  working checkpoint on a machine that hasn't trained anything, and at
  ~2.5MB it's not the burden that motivated gitignoring checkpoints
  generally. This isn't a policy change -- every other `.ckpt` stays
  gitignored.

## Demo

`src/app.py` -- a Gradio app, geometry in (0-3 holes, load), predicted
stress/displacement field out. Meshed live with `gmsh` at demo-time
(`src/mesh_gen.py`), not Abaqus -- a shareable demo can't depend on a
licensed Abaqus install plus the IIT Kanpur campus VPN. Uses the phase 11c2
checkpoint (the variable-hole-count model) deliberately, not the more
accurate single-hole specialist (phase 10b), because it has an actual
generalization story to show -- including its own honestly-reported limit,
surfaced as a visible in-app caveat when the user selects exactly 3 holes
(the held-out generalization test case, see `PROJECT_FLOW.md` phase 11c2).
Run with `python -m src.app` from the repo root (needs to run as a module,
not a script, for `src`'s own imports to resolve).

## Conventions

- Always report error as **relative L2 per field** (u_x, u_y, von_mises
  separately), never a single averaged number — same convention as AirfRANS.
- Don't tune architecture/hyperparameters before the data loader is solid and
  the model has been shown to overfit deliberately on a handful of cases.
- Local machine has a 4GB GPU (GTX 1650) — Abaqus solves are CPU-bound so
  dataset generation is unaffected, but keep local GNN training runs small;
  real training happens on Colab.

## Workflow

- Claude Code (this agent) handles multi-file/agentic work: the Abaqus
  generation/extraction scripts, mesh-to-graph converter, training loop,
  evaluation metrics, and any demo.
- GitHub Copilot (if active) handles inline autocomplete/boilerplate in the
  editor. Don't run both against the same file at the same time.
- Judging whether a resulting stress/displacement error is physically
  plausible is the user's call, not something to automate away.
