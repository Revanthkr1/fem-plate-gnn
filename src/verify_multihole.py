# -*- coding: mbcs -*-
# Run with: abaqus cae noGUI=src/verify_multihole.py
#
# Phase 11 verification: one hardcoded 2-hole case, before trusting the
# randomized multi-hole batch loop in generate_dataset.py. Confirms the
# multi-loop sketch/mesh/BC mechanics actually work in Abaqus -- same
# "verify the mechanism on a single case first" discipline as the original
# Kirsch check (phase 1) and first plasticity case (phase 2).
#
# Checks: job solves successfully, node count is sane, and -- the actual
# multi-hole-specific check -- no mesh nodes exist strictly inside either
# hole's circle (confirms both holes are real voids in the mesh, not one
# hole silently swallowing/ignoring the other).

import sys
import os

sys.path.insert(
    0, r'C:\Users\revan\OneDrive\Documents\Desktop\fem-plate-gnn\src')
from generate_dataset import (  # noqa: E402
    build_and_submit, job_completed_successfully,
)
from odbAccess import openOdb  # noqa: E402

LOG_PATH = 'verify_multihole.log'


def log(msg):
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


HOLES = [
    {'hole_r': 5.0, 'hole_x': 30.0, 'hole_y': 70.0},
    {'hole_r': 7.0, 'hole_x': 70.0, 'hole_y': 130.0},
]
PARAMS = {'holes': HOLES, 'load': 0.1}
JOB_NAME = 'Verify2Hole'
MODEL_NAME = 'Verify2HoleModel'


def main():
    log('Building and submitting 2-hole verification case: %r' % PARAMS)
    completed = build_and_submit(PARAMS, JOB_NAME, MODEL_NAME)
    log('job completed: %s' % completed)
    if not completed:
        log('FAIL: job did not complete -- leaving job files for inspection')
        return

    odb = openOdb(path=JOB_NAME + '.odb')
    instance = odb.rootAssembly.instances['PLATE-1']
    positions = [(n.coordinates[0], n.coordinates[1]) for n in instance.nodes]
    odb.close()

    log('node count: %d' % len(positions))

    violations = []
    for hole in HOLES:
        for (x, y) in positions:
            dist = ((x - hole['hole_x']) ** 2 + (y - hole['hole_y']) ** 2) ** 0.5
            if dist < hole['hole_r'] - 0.1:  # small tolerance for mesh curvature
                violations.append((hole, x, y, dist))

    if violations:
        log('FAIL: %d node(s) found strictly inside a hole (should be none):'
            % len(violations))
        for v in violations[:10]:
            log('  %r' % (v,))
    else:
        log('PASS: no mesh nodes found inside either hole -- both holes are '
            'real voids in the mesh.')


if __name__ == '__main__':
    main()
