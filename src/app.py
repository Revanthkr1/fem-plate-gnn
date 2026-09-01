"""Interactive Gradio demo: plate-with-hole geometry in, predicted stress field out.

Run with: python src/app.py

Uses the phase 11c2 checkpoint (data/model_release.ckpt) -- the
variable-hole-count model, chosen deliberately over the more accurate
single-hole specialist (phase 10b) because it's the one with an actual
generalization story to show, including its honestly-reported limit (see
PROJECT_FLOW.md phase 11c2).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import gradio as gr
import numpy as np

from src.mesh_gen import validate_holes, PLATE_W, PLATE_H, HOLE_MARGIN
from src.predict import predict
from src.data import load_case

CHECKPOINT_PATH = "data/model_release.ckpt"
STATS_PATH = "data/norm_stats.npz"

COUNT3_CAVEAT = (
    "**Note:** 3-hole configurations were held out of training entirely, "
    "specifically to test whether this model generalizes to a hole count "
    "it has never seen. It does reasonably well on stress *magnitude*, but "
    "peak-stress *location* accuracy is known to be substantially worse "
    "here (~48mm mean error vs. ~13mm for hole counts the model was "
    "trained on) -- see PROJECT_FLOW.md phase 11c2 for the full finding. "
    "This isn't hidden here on purpose."
)


def _plot_field(positions, elements, values, title):
    triangles = [e for e in elements.values() if len(e) == 3]
    # Split any quads into two triangles for plotting -- gmsh meshes here
    # are pure triangles, but keep this robust in case that ever changes.
    for e in elements.values():
        if len(e) == 4:
            triangles.append([e[0], e[1], e[2]])
            triangles.append([e[0], e[2], e[3]])
    tri = mtri.Triangulation(positions[:, 0], positions[:, 1], triangles)

    fig, ax = plt.subplots(figsize=(5, 8))
    contour = ax.tricontourf(tri, values, levels=20, cmap="viridis")
    fig.colorbar(contour, ax=ax, shrink=0.7)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.tight_layout()
    return fig


def run_prediction(n_holes, r1, x1, y1, r2, x2, y2, r3, x3, y3, load):
    all_holes = [
        {"hole_r": r1, "hole_x": x1, "hole_y": y1},
        {"hole_r": r2, "hole_x": x2, "hole_y": y2},
        {"hole_r": r3, "hole_x": x3, "hole_y": y3},
    ]
    holes = all_holes[: int(n_holes)]

    ok, msg = validate_holes(holes)
    if not ok:
        return None, f"**Invalid geometry:** {msg}", gr.update(visible=False)

    positions, elements, u_x, u_y, von_mises = predict(
        holes, load, checkpoint_path=CHECKPOINT_PATH, stats_path=STATS_PATH
    )
    fig = _plot_field(positions, elements, von_mises, "Predicted von Mises stress (MPa)")

    peak_idx = int(np.argmax(von_mises))
    info = (
        f"Predicted peak von Mises stress: **{von_mises[peak_idx]:.1f} MPa** "
        f"at ({positions[peak_idx, 0]:.1f}, {positions[peak_idx, 1]:.1f}) mm. "
        f"Mesh: {len(positions)} nodes, {len(elements)} elements."
    )
    caveat_visible = int(n_holes) == 3
    return fig, info, gr.update(visible=caveat_visible)


def update_hole_visibility(n_holes):
    n_holes = int(n_holes)
    return (
        gr.update(visible=n_holes >= 1),
        gr.update(visible=n_holes >= 2),
        gr.update(visible=n_holes >= 3),
    )


def load_example(case_id):
    """Preload a real held-out case's exact geometry, for a predicted-vs-
    actual comparison against ground truth already in data/raw/."""
    case = load_case("data/raw", case_id)
    holes = case["params"].get("holes")
    if holes is None:
        holes = [{
            "hole_r": case["params"]["hole_r"],
            "hole_x": case["params"]["hole_x"],
            "hole_y": case["params"]["hole_y"],
        }]
    load = case["params"]["load"]

    padded = holes + [{"hole_r": 8.0, "hole_x": 50.0, "hole_y": 100.0}] * (3 - len(holes))
    values = [len(holes)]
    for h in padded:
        values += [h["hole_r"], h["hole_x"], h["hole_y"]]
    values.append(load)
    return values


with gr.Blocks(title="FEM Plate-with-Hole GNN Surrogate") as demo:
    gr.Markdown(
        "# FEM Plate-with-Hole GNN Surrogate\n"
        "Predicts the displacement + von Mises stress field for a "
        f"{PLATE_W:.0f}x{PLATE_H:.0f}mm steel plate under uniaxial tension, "
        "with 0-3 circular holes, from geometry alone -- no hand-fed "
        "hole-location feature, just the mesh itself (see PROJECT_FLOW.md "
        "phase 10/10b). Meshed live with gmsh (not Abaqus, so this runs "
        "anywhere); the model was trained entirely on Abaqus-meshed data."
    )

    with gr.Row():
        with gr.Column(scale=1):
            n_holes = gr.Slider(0, 3, value=1, step=1, label="Number of holes")

            with gr.Group(visible=True) as hole1_group:
                gr.Markdown("**Hole 1**")
                r1 = gr.Slider(3.0, 15.0, value=8.0, label="Radius (mm)")
                x1 = gr.Slider(HOLE_MARGIN, PLATE_W - HOLE_MARGIN, value=50.0, label="X (mm)")
                y1 = gr.Slider(HOLE_MARGIN, PLATE_H - HOLE_MARGIN, value=100.0, label="Y (mm)")

            with gr.Group(visible=False) as hole2_group:
                gr.Markdown("**Hole 2**")
                r2 = gr.Slider(3.0, 15.0, value=6.0, label="Radius (mm)")
                x2 = gr.Slider(HOLE_MARGIN, PLATE_W - HOLE_MARGIN, value=30.0, label="X (mm)")
                y2 = gr.Slider(HOLE_MARGIN, PLATE_H - HOLE_MARGIN, value=60.0, label="Y (mm)")

            with gr.Group(visible=False) as hole3_group:
                gr.Markdown("**Hole 3**")
                r3 = gr.Slider(3.0, 15.0, value=5.0, label="Radius (mm)")
                x3 = gr.Slider(HOLE_MARGIN, PLATE_W - HOLE_MARGIN, value=70.0, label="X (mm)")
                y3 = gr.Slider(HOLE_MARGIN, PLATE_H - HOLE_MARGIN, value=150.0, label="Y (mm)")

            load = gr.Slider(0.05, 0.20, value=0.10, label="Applied top displacement (mm)")

            predict_btn = gr.Button("Predict", variant="primary")

            gr.Markdown("**Or load a real held-out example:**")
            with gr.Row():
                ex_indist_btn = gr.Button("Example: in-distribution (2 holes)")
                ex_ood_btn = gr.Button("Example: held-out (3 holes)")

        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Predicted von Mises stress")
            info_output = gr.Markdown()
            caveat_output = gr.Markdown(COUNT3_CAVEAT, visible=False)

    n_holes.change(
        update_hole_visibility, inputs=[n_holes],
        outputs=[hole1_group, hole2_group, hole3_group],
    )

    all_inputs = [n_holes, r1, x1, y1, r2, x2, y2, r3, x3, y3, load]
    predict_btn.click(run_prediction, inputs=all_inputs,
                       outputs=[plot_output, info_output, caveat_output])

    ex_indist_btn.click(lambda: load_example(207), outputs=all_inputs).then(
        run_prediction, inputs=all_inputs, outputs=[plot_output, info_output, caveat_output]
    )
    # A real held-out count=3 case, per src/splits.py's split (case ids
    # with exactly 3 holes were never used in training).
    ex_ood_btn.click(lambda: load_example(202), outputs=all_inputs).then(
        run_prediction, inputs=all_inputs, outputs=[plot_output, info_output, caveat_output]
    )


if __name__ == "__main__":
    demo.launch()
