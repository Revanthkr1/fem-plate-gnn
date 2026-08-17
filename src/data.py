"""Load one cached plate-with-hole case (written by src/generate_dataset.py)."""
import json
import os


def load_case(raw_dir, case_id):
    """Returns the raw case dict: {'case_id', 'params', 'nodes', 'elements',
    'displacement', 'von_mises'} -- see src/generate_dataset.py::extract_case
    for exactly how each field was written from the Abaqus odb.
    """
    path = os.path.join(raw_dir, "case_%03d.json" % case_id)
    with open(path, "r") as f:
        return json.load(f)
