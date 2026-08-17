# -*- coding: mbcs -*-
# Run with: abaqus python src/check_kirsch.py <path-to-odb>
#
# Phase 1 sanity check: compares the FE stress concentration factor (SCF)
# against the analytical Kirsch solution for a small circular hole in a plate
# under uniaxial tension (SCF -> 3 as hole radius / plate width -> 0).

import sys
from odbAccess import openOdb

PLATE_W = 100.0
THICKNESS = 1.0
EXPECTED_SCF = 3.0
TOLERANCE = 0.10  # allow 10% deviation from the ideal infinite-plate SCF


def main(odb_path):
    odb = openOdb(path=odb_path)
    step = odb.steps['Load']
    frame = step.frames[-1]

    stress_field = frame.fieldOutputs['S']
    max_mises = 0.0
    for value in stress_field.values:
        if value.mises > max_mises:
            max_mises = value.mises

    rf_field = frame.fieldOutputs['RF']
    bottom_set = odb.rootAssembly.nodeSets['BOTTOM_EDGE_SET']
    rf_subset = rf_field.getSubset(region=bottom_set)
    total_rf2 = sum(v.data[1] for v in rf_subset.values)

    nominal_stress = abs(total_rf2) / (PLATE_W * THICKNESS)
    scf = max_mises / nominal_stress

    print('Max von Mises stress:   %.4f' % max_mises)
    print('Nominal (far-field) stress: %.4f (from reaction force sum)' % nominal_stress)
    print('FE stress concentration factor: %.4f' % scf)
    print('Analytical (Kirsch) SCF:        %.4f' % EXPECTED_SCF)

    rel_error = abs(scf - EXPECTED_SCF) / EXPECTED_SCF
    print('Relative error vs analytical: %.2f%%' % (rel_error * 100.0))
    if rel_error <= TOLERANCE:
        print('PASS: within %.0f%% of the analytical Kirsch SCF.' % (TOLERANCE * 100.0))
    else:
        print('FAIL: outside %.0f%% tolerance -- check mesh refinement / BCs.' % (TOLERANCE * 100.0))

    odb.close()


if __name__ == '__main__':
    odb_arg = sys.argv[-1]
    main(odb_arg)
