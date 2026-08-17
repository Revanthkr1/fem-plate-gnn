"""Turn one plate-with-hole case into a graph: node features + edge_index + edge features.

Mirrors the AirfRANS project's src/graph.py pattern (bidirectional mesh edges via
PyVista's extract_all_edges(), edge_attr = relative position) -- the mesh here
comes from Abaqus (CPS4/CPS4R quads, with a few CPS3 triangles from the
quad-dominated free mesher around the hole) instead of gmsh/FEniCSx, but the
same PyVista call handles either once the connectivity is loaded into an
UnstructuredGrid.
"""
import numpy as np
import pyvista as pv


def build_graph(case):
    """
    Node features: [x, y, signed_distance_to_hole, load_magnitude]  (N, 4)
    Edges: element-edge connectivity, made bidirectional.
    Edge features: relative position (dst - src)  (2*E, 2)
    Targets: [u_x, u_y, von_mises]  (N, 3)

    `case` is the dict returned by src/data.py::load_case (one case_NNN.json).
    """
    labels = sorted(case["nodes"].keys(), key=int)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    position = np.array([case["nodes"][label] for label in labels], dtype=np.float64)

    params = case["params"]
    hole_center = np.array([params["hole_x"], params["hole_y"]])
    signed_distance = np.linalg.norm(position - hole_center, axis=1) - params["hole_r"]
    load_magnitude = np.full(len(labels), params["load"])

    node_features = np.concatenate(
        [position, signed_distance[:, None], load_magnitude[:, None]], axis=1
    )

    displacement = np.array([case["displacement"][label] for label in labels])
    von_mises = np.array([case["von_mises"][label] for label in labels])
    targets = np.concatenate([displacement, von_mises[:, None]], axis=1)

    edge_index, edge_attr = _mesh_edges(case["elements"], label_to_idx, position)

    return node_features, edge_index, edge_attr, targets


def _mesh_edges(elements, label_to_idx, position):
    cells_flat = []
    cell_types = []
    for elem_nodes in elements.values():
        idx = [label_to_idx[str(n)] for n in elem_nodes]
        cells_flat.append(len(idx))
        cells_flat.extend(idx)
        cell_types.append(pv.CellType.QUAD if len(idx) == 4 else pv.CellType.TRIANGLE)

    points_3d = np.concatenate([position, np.zeros((len(position), 1))], axis=1)
    grid = pv.UnstructuredGrid(np.array(cells_flat), np.array(cell_types), points_3d)

    edges_poly = grid.extract_all_edges()
    pairs = edges_poly.lines.reshape(-1, 3)[:, 1:]  # (E, 2), format [n_pts, i, j] per cell

    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    edge_index = np.stack([src, dst], axis=0)

    edge_attr = position[dst] - position[src]

    return edge_index, edge_attr
