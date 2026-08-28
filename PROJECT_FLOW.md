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

## Phase 8 — First full training run + evaluation (done, mixed result)

Trained on Colab: all 200 cases, 180/20 train/val split, `configs/base.yaml`
hyperparameters (latent_dim=32, hidden_dim=64, n_message_passing=4,
max_epochs=100), plain MSE loss. Checkpoint verified against the saved
`state_dict` (`node_encoder` input dim 4, `global_step=4500` = 180 cases / 4
accum steps * 100 epochs, exactly as expected) before evaluating -- confirmed
this really is the fully-trained FEM checkpoint, not an AirfRANS one (both
projects' checkpoints share the same `mgn-epoch=NNN.ckpt` naming).

Wrote `src/evaluate.py` (ported from AirfRANS's pattern) and ran it on the
20 held-out cases (ids 180-199):

| Field | Relative L2 | Mean abs error |
|---|---|---|
| u_x | 38% (8-131% per-case spread) | 0.0031 mm |
| u_y | 4.8% | 0.0027 mm |
| von_mises | 13.5% | 10.8 MPa |

Peak-stress sanity check (this project's analogue of AirfRANS's Cd/Cl check --
literal Kirsch SCF doesn't apply per-case once plasticity is active):
**peak von Mises relative error 37%, mean peak-location miss 18.1mm** (on a
100x200mm plate with 3-15mm hole radii).

**Reading these numbers**: u_x's huge relative-L2 spread is a metric
artifact, not a real weakness -- its absolute error (0.0031mm) is essentially
identical to u_y's (0.0027mm), but u_x (lateral Poisson-effect displacement)
is 6-7x smaller in magnitude than u_y (primary load-direction displacement)
in every case checked, so the same absolute error divides into a much bigger
relative number. Exactly the near-zero-denominator pathology
`metrics.py::relative_l2_per_field`'s own docstring warns about --
`mean_abs_error_per_field` is the right lens for this field, not something to
chase by changing the model.

The genuine concern is the peak-stress localization: the bulk von Mises field
is decent (13.5% relative L2), but the model isn't reliably finding *where*
or *how bad* the worst-case stress is -- the number that matters most for an
actual structural surrogate, since it's what predicts failure. First full
pass through the pipeline works end-to-end, but isn't yet trustworthy for
that specific job. Candidate next steps (not yet decided): more epochs,
more model capacity, a peak-stress-weighted loss (mirroring the AirfRANS
project's wall-distance-weighted loss precedent), or more training data.

## Phase 9 — Retrain to 250 epochs (result muddied by a checkpoint-collision bug -- see phase 10)

Cheapest candidate fix tried first: bumped `max_epochs` 100 -> 250 (fresh
run, new checkpoint path `meshgraphnet_250ep.ckpt` -- see `configs/base.yaml`
and both notebooks' section-5/4 markdown for why resuming the old checkpoint
would have corrupted the Cosine-annealing schedule instead of cleanly
isolating this one variable). Checkpoint verified the same way as the first
run (`node_encoder` input dim 4, `epoch=249`, `global_step=11250` = 180
cases / 4 accum steps * 250 epochs, exact match) before evaluating.

Same 20 held-out cases (180-199), same `src/evaluate.py`:

| Metric | 100 epochs | 250 epochs |
|---|---|---|
| u_x rel L2 | 37.8% | 39.6% (unchanged -- metric artifact, see phase 8) |
| u_y rel L2 | 4.8% | 5.6% (unchanged) |
| von_mises rel L2 (bulk) | 13.5% | 12.5% (unchanged) |
| **peak von_mises rel L2** | **37.1%** | **6.0%** |
| **mean peak-location error** | **18.1mm** | **9.8mm** |

**Result: peak-stress accuracy improved a lot, but the experiment wasn't as
clean as reported at the time.** Bulk field accuracy barely moved between
runs, while peak-stress accuracy improved 6x (37% -> 6% relative error) and
location error roughly halved. Original writeup called this a controlled
"100 vs 250 epochs, fresh run" comparison -- **that part turned out to be
wrong.**

Caught during phase 10 (see below): `src/train.py`'s auto-resume
glob-searches the whole *directory* a checkpoint lives in for any
`mgn-epoch=*.ckpt` file, not just the specific final filename. Phase 8's
`meshgraphnet.ckpt` and phase 9's `meshgraphnet_250ep.ckpt` both lived in the
same flat `DRIVE_ROOT` folder on Drive -- changing the final filename didn't
stop the auto-resume from finding phase 8's periodic `mgn-epoch=099.ckpt` and
silently resuming from it (no crash, since both runs shared the same
4-feature architecture -- the mismatch only became obvious once phase 10
changed `node_in_dim` and the *same* collision produced a loud shape-mismatch
error instead of a silent resume). Inspecting the actual phase-9 checkpoint
confirms it: `lr_schedulers[0]['T_max']` is **100**, not 250 -- the Cosine
schedule had already annealed to 0 by epoch 100 during phase 8, then phase 9
continued 150 more epochs with `T_max` still stuck at 100 (not rebuilt for
the new `max_epochs=250`), so the back half of "phase 9" trained under a
cyclical/re-warming LR pattern rather than the intended smooth 250-epoch
anneal.

**Honest read**: the model did receive substantially more cumulative
training (350 total epochs across both runs, continued rather than
independent), so "more training helps peak-stress accuracy" is probably
still directionally true -- but this was not the clean, isolated
100-vs-250-epochs comparison it was described as. A genuinely clean rerun
(fresh weights, real 250-epoch schedule, own checkpoint subdirectory) would
be needed to actually confirm the undertraining hypothesis in isolation;
not yet decided whether that's worth doing given the current result is
already usable.

## Phase 10 — Non-parametric ablation: drop the hand-engineered geometry feature (in progress)

Motivated by researching Neural Concept for context: their surrogate is
**non-parametric** (trains directly on raw mesh geometry, no hand-crafted
parameterization), while ours was being handed the hole's exact
center/radius as a feature (`signed_distance_to_hole`) instead of learning
proximity effects from the mesh itself. Full plan (this phase + phase 11,
variable hole count, the actual generalization test) in
`.claude/plans/optimized-gliding-cookie.md`.

Changes: `src/graph.py::build_graph()` no longer reads hole geometry from
`params` at all -- `node_features` is now `[x, y, load]` (3, down from 4).
`configs/base.yaml` (`node_in_dim: 3`), `data/norm_stats.npz` recomputed,
local `data/cache/` regenerated, both notebooks updated to match.

**Bug found and fixed while setting up this run's retraining** (this is the
same bug that muddied phase 9, see above): `src/train.py`'s auto-resume
glob-searches the *entire directory* a checkpoint path lives in for any
`mgn-epoch=*.ckpt` file -- changing only the final checkpoint's filename
(what phase 9 did) doesn't prevent collision with periodic checkpoints from
a different run sitting in the same directory. This phase's attempt crashed
loudly with a shape mismatch (`[64, 4]` vs `[64, 3]`) trying to resume into
phase 9's checkpoint, which is what surfaced the bug -- phase 9's version of
the same collision didn't crash (same 4-feature architecture both times) and
went unnoticed until now. **Fix**: every experiment now gets its own
checkpoint *subdirectory* (e.g. `DRIVE_ROOT/nogeomfeat/`), not just a
different filename in a shared directory -- applied to both notebooks.

**Result: the gate did not pass -- dropping the feature caused a real
regression.** Checkpoint verified clean this time (`node_encoder` input dim
3, `epoch=250`, `global_step=11250`, and critically `lr_schedulers[0].T_max
== 250` -- confirming this really was a fresh, isolated 250-epoch run, not
another silent resume). Evaluated on the same 20 held-out cases:

| Metric | Phase 8 (100ep, clean, **with** feature) | Phase 10 (250ep, clean, **without** feature) |
|---|---|---|
| u_y rel L2 | 4.8% | 4.5% (unchanged) |
| von_mises rel L2 (bulk) | 13.5% | 14.5% (unchanged) |
| peak von_mises rel L2 | 37.1% | 37.4% (unchanged) |
| **mean peak-location error** | **18.1mm** | **51.6mm** |

Even with 2.5x the training of the clean baseline, removing
`signed_distance_to_hole` left peak-*magnitude* accuracy about the same but
made peak-*location* accuracy dramatically worse (worst location error seen
in the project so far). Useful negative result, not a dead end: it isolates
what the feature was actually doing -- helping the model know **where** to
expect the concentration, not how big it gets. Plausible root cause: with no
explicit distance feature, the model's only path to "where's the hole" is
message-passing hops from local mesh curvature, and `n_message_passing=4`
(`configs/base.yaml`) gives each node only 4 hops of receptive field --
nowhere near enough for a distant node to infer global hole-relative
position purely from propagated local curvature cues.

Per the plan's explicit gate, this blocked moving straight to phase 11
(variable hole count) without addressing it first. Candidates considered:
(a) increase `n_message_passing`; (b) a topology-native distance feature
(BFS hop-count to nearest hole boundary, no hole_r/x/y needed); (c) accept
the with-feature version and treat non-parametric as a stretch goal. User
picked (a) first, as the cheapest to test.

## Phase 10b — More message-passing hops: gate passed, non-parametric now wins outright

`n_message_passing` 4 -> 8 (`configs/base.yaml`), same 3-feature
(`[x, y, load]`) representation, same 250-epoch budget, own checkpoint
subdirectory (`.../moremp/`, learned from the phase-9 collision). Checkpoint
verified clean: `node_encoder` input dim 3, 8 message-passing blocks (up
from 4), `epoch=249`, `global_step=11250`, `lr_schedulers[0].T_max == 250`.

| Model | Features | Peak von Mises rel L2 | Peak-location error |
|---|---|---|---|
| Phase 8 (100ep, clean, **with** feature) | 4 | 37.1% | 18.1mm |
| Phase 9 (muddied, ~350 cumulative epochs, **with** feature) | 4 | 6.0% | 9.8mm |
| Phase 10 (250ep clean, **no** feature, 4 hops) | 3 | 37.4% | 51.6mm |
| **Phase 10b (250ep clean, no feature, 8 hops)** | 3 | **3.9%** | **4.9mm** |

**Result: hypothesis confirmed, and then some.** The fully non-parametric
model (no hand-fed hole geometry at all) now beats every previous result,
including the best hand-engineered-feature run, on every metric -- bulk
`von_mises` also improved slightly (11.9% vs. 12.5-14.5% in earlier runs).
Confirms receptive field, not the missing feature itself, was the real
bottleneck: once nodes have enough hops to propagate hole-boundary signal
(finer local mesh spacing near holes is self-detectable, no external hint
needed), the model learns to both find and size the stress concentration
better than when it was handed the answer directly. This is the actual
"Neural Concept style" result being aimed for -- geometry generalization
learned from the mesh, not from hand parameterization.

This is now the base to build phase 11 (variable hole count) on top of.

**Full take-stock evaluation** (all 200 cases, train vs. held-out split,
before starting phase 11) -- `data/eval_full_200_moremp.json` has full
per-case detail:

| Metric | Train (180 cases) | Held-out (20 cases) |
|---|---|---|
| u_y rel L2 | 4.6% | 4.6% |
| von_mises rel L2 (bulk) | 11.4% | 11.9% |
| peak von_mises rel L2 | 4.1% | 3.9% |
| mean peak-location error | 7.5mm | 4.9mm |

Train and held-out are nearly identical across every metric -- held-out is
even marginally *better* on two of them, well within noise. This rules out
the model simply memorizing training cases; it's a genuinely well-calibrated
fit at this dataset size, not an overfit one that would look great on train
and fall apart on new data.

## Phase 11 — Variable hole count: data generation (done)

Extended `src/generate_dataset.py` for the actual geometric-generalization
test: hole *count* now varies (0-3), not just radius/position within one
template. `sample_params()` draws a target count, then rejection-samples
each hole's `(r, x, y)` (retry on edge-margin violation or overlap with an
already-placed hole, capped at 50 attempts/hole, falls back to fewer holes
and logs it rather than crashing -- never triggered in the actual run, 0
fallbacks across 200 cases). `build_and_submit()` loops the sketch/mesh-
seeding per hole; 0 holes is just the rectangle with the loop skipped, not a
special case. `params` format changes to `{"holes": [...], "load": ...}` --
not a breaking change in practice, since `build_graph()` (post phase 10)
never reads hole geometry from `params`, only `params["load"]`.

**Verified on one hardcoded 2-hole case first** (`src/verify_multihole.py`)
before trusting the randomized batch: job solved, 8589 nodes, and critically
**no mesh nodes found inside either hole** -- confirms both holes are real
voids in the mesh (the multi-loop sketch mechanism works), not one silently
swallowing the other.

Generated cases 200-399: **200/200 succeeded, 0 failures, 0 placement
fallbacks**. Hole-count distribution: 0 holes=57, 1 hole=46 (+200 from the
original single-hole cases = 246 total), 2 holes=46, 3 holes=51.

**The actual generalization split** (`src/splits.py::hole_count_splits()`):
holds out every count=3 case (51 total) from training entirely -- the real
test of whether the model generalizes to a hole count it has never seen, not
just a new radius/position within a count it has seen. Remaining 349 cases
(counts 0/1/2) split into 329 train + 20 in-distribution validation (random,
seeded). `norm_stats.npz` recomputed over the 349 in-distribution cases only
-- excludes the held-out count=3 bucket so normalization can't leak any
information about it.

Architecture carried over unchanged from phase 10b (`node_in_dim=3`,
`n_message_passing=8` -- no hand-fed hole geometry, the validated
non-parametric setup). Own checkpoint subdirectory (`.../phase11/`), same
collision-avoidance discipline as every run since phase 9.

## Phase 11b — Training + generalization evaluation (done, honest mixed result)

Checkpoint verified clean: `node_encoder` input dim 3, 8 message-passing
blocks, `epoch=249`, `T_max=250` (fresh run), `global_step=20750` = 329
train cases / 4 accum steps * 250 epochs exactly.

Evaluated on both splits with `src/evaluate.py`. Along the way, found and
fixed a real bug in the evaluation code itself: 0-hole cases have no actual
stress concentration (the field is smooth/near-uniform under uniaxial
tension), so their "true peak location" is essentially arbitrary mesh noise
-- including them in the peak-location-error average was comparing noise
against noise and inflating the number in a way that looked like model
failure but wasn't. `summarize()` now excludes 0-hole cases from that metric
specifically (peak magnitude and bulk field errors are still meaningful for
them, so those stay computed over the full set) -- see `src/evaluate.py`.

Corrected numbers (cases with >=1 hole only, for the location metric):

| Split | Peak magnitude error | Peak location error (mean / median) |
|---|---|---|
| Phase 10b (single-hole only, for reference) | 3.9% | 4.9mm / -- |
| **In-distribution held-out** (0/1/2 holes, 16 with-hole cases) | 31.6% | 43.9mm / 32.1mm |
| **OOD held-out** (count=3, never seen, 51 cases) | 33.2% | 49.0mm / 44.9mm |

**Honest read, two separate findings:**

1. **The actual point of phase 11 -- the generalization gap -- is small.**
   In-distribution (31.6%/43.9mm) vs. genuinely-unseen count=3 (33.2%/49.0mm)
   are close. The model generalizes to a hole count it never trained on
   almost as well as it performs on counts it did train on. That's the real
   demonstration this phase was built to produce, and it holds up.

2. **But both numbers are much worse than phase 10b's single-hole-only
   result.** Training on a more geometrically diverse dataset (variable hole
   count, not just one template) is a harder learning problem -- the model
   now has to handle a variable number of candidate stress concentrations
   and correctly judge which one (if any) dominates, not just regress a
   single hole's effect. The same 250-epoch budget and architecture that
   fully solved the easier single-hole problem hasn't yet reached the same
   quality on this harder one.

Not yet decided: whether to push training further (more epochs, given
undertraining was exactly the phase-10b lesson for a harder localization
problem), add capacity, or treat "small generalization gap, weaker absolute
accuracy" as the honest, sufficient conclusion for this phase and move on.
User picked more epochs.

## Phase 11c — More epochs: muddied by the phase-9 bug recurring, fixed properly this time

Bumped `max_epochs` 250 -> 500 (`configs/base.yaml`), new checkpoint
subdirectory (`.../phase11c/`). The resulting checkpoint (`epoch=499`,
`global_step=41500` -- consistent with either a clean 500-epoch run OR a
resumed 250-then-continued-250-more run, since both land on the same total)
had **`lr_schedulers[0].T_max == 250`, not 500** -- the exact phase-9 bug
recurring, despite the new subdirectory. `global_step` alone could not have
caught this; only inspecting the checkpoint's own saved scheduler state did.

**Root-caused and fixed properly this time**, not just via directory
hygiene (which had now failed twice): added
`src/train.py::_check_resume_schedule_compatibility()`, called right before
`trainer.fit()` whenever resuming -- compares the checkpoint's saved
scheduler `T_max` against the requested `max_epochs` and raises immediately
if they don't match, instead of silently corrupting the schedule via
`load_state_dict`. Verified against the actual mismatched checkpoint
(correctly raises) and a matching case (correctly stays silent). This is a
permanent guard against the whole bug class, not a one-off patch -- any
future mismatched resume now fails loudly at start, not silently months
later when someone happens to inspect a checkpoint's internals.

Phase 11c needs re-running clean before its result can be trusted.

## Phase 11c2 — Clean 500-epoch retry: reveals the earlier "small gap" was a floor effect

Re-ran in a fresh subdirectory (`phase11c2/`), protected by the new guard.
Checkpoint verified clean: `T_max=500` (matches `max_epochs`), `epoch=499`,
`global_step=41500` (= 83 steps/epoch * 500 epochs, consistent).

| Split | Peak magnitude error | Peak location error |
|---|---|---|
| Phase 11b (250ep, muddied) in-distribution | 31.6% | 43.9mm |
| Phase 11b (250ep, muddied) OOD (count=3) | 33.2% | 49.0mm |
| **Phase 11c2 (500ep, clean) in-distribution** | **4.2%** | **13.1mm** |
| **Phase 11c2 (500ep, clean) OOD (count=3)** | **6.4%** | **47.8mm** |

**This reframes phase 11b's conclusion.** More training fixed
in-distribution accuracy dramatically (31.6%->4.2%, 43.9mm->13.1mm, now
close to phase 10b's single-hole-specialist quality). But it barely moved
the OOD location error (49.0mm -> 47.8mm, essentially flat), even though
OOD magnitude error did improve (33.2%->6.4%). The real generalization gap
is now exposed: 13.1mm in-distribution vs. 47.8mm on truly unseen hole
count -- **~3.6x worse**, not the "small gap" phase 11b reported.

Phase 11b's small in-dist-vs-OOD gap was a floor effect, not evidence of
good generalization: undertaining was making everything look similarly bad
(43.9mm and 49.0mm aren't that different when both are far from converged),
which masked a real weakness underneath. Once in-distribution actually
converged, the gap to genuinely unseen geometry became obvious. More
training teaches the model to do very well on hole counts it has seen; it
does not, by itself, teach it to transfer that precision to a hole count it
has never seen -- that's a harder, structurally different problem
(learning to generalize across a discrete structural variable, not just
interpolating within a continuous parameter range like radius/position).

This is a more rigorous and more interesting finding than the original
report, even though less flattering -- catching that an apparently good
result was actually an undertraining artifact is exactly the kind of
verification this project's evaluation discipline is supposed to produce.

## Remaining phases

13. Decide how to respond to the real generalization gap phase 11c2
    exposed (13.1mm in-distribution vs. 47.8mm on unseen hole count).
    Candidates: more training data spanning more hole counts so "count" is
    less of a discrete jump; an explicit count-agnostic architectural
    change (e.g. attention/pooling over per-hole local signals rather than
    plain message-passing); or accepting this as the honest limit of the
    current approach and documenting it as a finding, not a fix-it item.
14. Decide whether to also do a genuinely clean rerun of the phase-9
    comparison (fresh weights, isolated checkpoint directory) to confirm the
    undertraining hypothesis in isolation -- open since phase 9, low
    priority now that phase 10b's result stands on its own regardless.
15. "Copilot" demo (geometry/parameters in, instant prediction out via the
    trained GNN — mirroring how Neural Concept's actual product works, not a
    natural-language agent).
16. README narrative, and — if desired — a PhysicsNeMo v2 port, mirroring
    the AirfRANS project's stated ambition.
