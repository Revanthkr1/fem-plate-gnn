"""Demo-time mesh generation, independent of Abaqus.

The trained model predicts on a mesh graph (node positions + element
connectivity), not on raw geometry parameters -- so a user-specified hole
layout has to become an actual mesh before the model can run on it. Using
Abaqus for this at demo-time isn't viable (needs a licensed install + the
IIT Kanpur campus VPN, which nobody else running this demo would have), so
this uses gmsh instead. The model itself is agnostic to element type (quad
vs triangle -- src/graph.py's _mesh_edges() already handles Abaqus's own
mixed quad/triangle meshes via generic element-to-edge extraction), so a
pure-triangle gmsh mesh is fine -- no need to replicate Abaqus's specific
quad-dominated meshing algorithm, just similar density characteristics.
"""
import gmsh
import numpy as np

PLATE_W = 100.0
PLATE_H = 200.0
HOLE_MARGIN = 12.0  # min distance from hole edge to any plate boundary
MIN_HOLE_GAP = 5.0  # mm, minimum gap between two holes' boundaries
GLOBAL_MESH_SIZE = 2.0
HOLE_MESH_SIZE_FACTOR = 0.15  # local size near a hole = hole_r * this factor


def validate_holes(holes, plate_w=PLATE_W, plate_h=PLATE_H,
                    margin=HOLE_MARGIN, min_gap=MIN_HOLE_GAP):
    """Same geometric constraints src/generate_dataset.py's rejection
    sampling enforces (edge margin, pairwise hole gap) -- reimplemented
    standalone since that file only runs under Abaqus's own Python 2.7
    kernel and can't be imported here. Returns (ok, message)."""
    for i, hole in enumerate(holes):
        r, x, y = hole["hole_r"], hole["hole_x"], hole["hole_y"]
        if x - r < margin or x + r > plate_w - margin:
            return False, f"Hole {i + 1}: too close to the left/right edge (need >= {margin}mm margin)."
        if y - r < margin or y + r > plate_h - margin:
            return False, f"Hole {i + 1}: too close to the top/bottom edge (need >= {margin}mm margin)."
        for j, other in enumerate(holes[:i]):
            dist = ((x - other["hole_x"]) ** 2 + (y - other["hole_y"]) ** 2) ** 0.5
            if dist < r + other["hole_r"] + min_gap:
                return False, f"Holes {j + 1} and {i + 1} overlap or are too close (need >= {min_gap}mm gap)."
    return True, ""


def generate_mesh(holes, plate_w=PLATE_W, plate_h=PLATE_H):
    """Mesh a plate_w x plate_h rectangle with the given circular holes cut
    out. Returns (positions, elements): positions is (N, 2) float array;
    elements is {label: [node labels]} in the same shape
    src/graph.py::_mesh_edges() already expects, so that function is reused
    unchanged rather than rewriting edge extraction.
    """
    # interruptible=False: gmsh.initialize() otherwise installs a SIGINT
    # handler via signal.signal(), which only works in the main thread of
    # the main interpreter -- this is called from Gradio request callbacks,
    # which run in worker threads, so the default raises
    # "ValueError: signal only works in main thread of the main interpreter".
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("plate")

        rect = gmsh.model.occ.addRectangle(0, 0, 0, plate_w, plate_h)
        if holes:
            disks = [
                gmsh.model.occ.addDisk(h["hole_x"], h["hole_y"], 0, h["hole_r"], h["hole_r"])
                for h in holes
            ]
            gmsh.model.occ.cut([(2, rect)], [(2, d) for d in disks])
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMax", GLOBAL_MESH_SIZE)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.3)

        if holes:
            # Refine near each hole boundary via a Distance+Threshold field --
            # find each hole's boundary curve by checking which curve's
            # bounding-box center sits ~hole_r from that hole's center
            # (outer rectangle edges are far from any hole center; a hole's
            # own boundary circle is not).
            curves = gmsh.model.getEntities(dim=1)
            field_ids = []
            for h in holes:
                for (dim, tag) in curves:
                    bbox = gmsh.model.getBoundingBox(dim, tag)
                    cx, cy = (bbox[0] + bbox[3]) / 2, (bbox[1] + bbox[4]) / 2
                    dist = ((cx - h["hole_x"]) ** 2 + (cy - h["hole_y"]) ** 2) ** 0.5
                    if dist < h["hole_r"] * 0.5:  # curve is centered near this hole
                        dist_field = gmsh.model.mesh.field.add("Distance")
                        gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", [tag])
                        thresh_field = gmsh.model.mesh.field.add("Threshold")
                        gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
                        gmsh.model.mesh.field.setNumber(
                            thresh_field, "SizeMin", max(h["hole_r"] * HOLE_MESH_SIZE_FACTOR, 0.3))
                        gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", GLOBAL_MESH_SIZE)
                        gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", h["hole_r"] * 0.5)
                        gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", h["hole_r"] * 2.0)
                        field_ids.append(thresh_field)
                        break
            if field_ids:
                min_field = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", field_ids)
                gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.array(node_tags, dtype=np.int64)
        coords = np.array(node_coords, dtype=np.float64).reshape(-1, 3)[:, :2]

        # gmsh node tags are 1-indexed and not guaranteed contiguous -- map
        # to 0-based contiguous labels matching positions' row order.
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
        positions = coords

        elem_types, _, elem_node_tags_list = gmsh.model.mesh.getElements(dim=2)
        elements = {}
        label = 0
        for etype, node_tags_flat in zip(elem_types, elem_node_tags_list):
            n_nodes_per_elem = 3 if etype == 2 else (4 if etype == 3 else None)
            if n_nodes_per_elem is None:
                continue  # skip any non-tri/quad element type
            node_tags_flat = np.array(node_tags_flat, dtype=np.int64).reshape(-1, n_nodes_per_elem)
            for elem_nodes in node_tags_flat:
                elements[label] = [tag_to_idx[int(t)] for t in elem_nodes]
                label += 1

        return positions, elements
    finally:
        gmsh.finalize()
