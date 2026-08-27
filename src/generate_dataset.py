# -*- coding: mbcs -*-
# Run with: abaqus cae noGUI=src/generate_dataset.py
#
# Phase 3 (cases 0-199): parametrized batch generation, exactly one circular
# hole per case, hole radius/position/load randomized. Phase 11 (cases
# 200+): variable hole COUNT (0-3) via rejection-sampled placement -- the
# actual geometric-generalization test, not just interpolating within one
# template. Both share the same loop: draws random params, solves each case
# through a single CAE session (avoids the ~15-20s per-invocation startup
# cost of spawning `abaqus cae` N separate times), extracts mesh + fields to
# a compact per-case JSON, and discards the heavy Abaqus job artifacts
# (.odb/.sta/.msg/...) to save disk, mirroring the AirfRANS project's
# cache-then-discard convention.

import json
import os
import random
import traceback

from abaqus import mdb
from abaqusConstants import (
    TWO_D_PLANAR, DEFORMABLE_BODY, OFF, ON, STANDARD, CPS4R, FINER,
    ELEMENT_NODAL,
)
from caeModules import *
import regionToolset
from mesh import ElemType
from odbAccess import openOdb

PLATE_W = 100.0
PLATE_H = 200.0
YOUNGS_MODULUS = 210000.0
POISSONS_RATIO = 0.3
PLASTIC_TABLE = ((250.0, 0.0), (300.0, 0.01), (400.0, 0.05))
MESH_SIZE = 2.0
MESH_SIZE_HOLE_FACTOR = 0.15  # local seed size = hole radius * this factor

# Cases 0-199 (single hole, hole_r/x/y varying) already generated and
# trained on -- see PROJECT_FLOW.md phases 3-10b. Phase 11 continues from
# case 200: variable hole COUNT (0-3), the actual geometric-generalization
# test, not just interpolating within one template.
START_CASE = 200
N_CASES = 200  # ~50 cases/bucket across counts {0,1,2,3} -- enough to train
               # on {0,1,2} and hold out an entire unseen count (3) as the
               # generalization test, per the phase-11 plan, while the
               # original 200 single-hole cases add extra data to the
               # count=1 bucket.
RANDOM_SEED = 0  # combined with case_id per-case (see sample_params) -- a
                 # given case_id always draws the same params regardless of
                 # what range of cases a given run covers.
HOLE_R_RANGE = (3.0, 15.0)
HOLE_MARGIN = 12.0  # min distance from hole edge to any plate boundary
LOAD_RANGE = (0.05, 0.20)  # mm, top edge displacement
HOLE_COUNT_RANGE = (0, 3)  # inclusive both ends -- randint draws 0,1,2, or 3
MIN_HOLE_GAP = 5.0  # mm, minimum gap between two holes' boundaries
MAX_PLACEMENT_ATTEMPTS = 50  # per hole, before falling back to fewer holes
                              # for that case (logged, not a crash -- same
                              # "log and skip" convention as non-converging
                              # Abaqus jobs)

INSTANCE_NAME = 'Plate-1'
OUTPUT_DIR = os.path.join('..', '..', 'raw')  # data/runs/<batch> -> data/raw
BB_TOL = 1e-3
LOG_PATH = 'generate_dataset.log'


def log(msg):
    # abaqus cae noGUI does not relay Python print() output to the invoking
    # shell, so progress/errors are written to a log file instead.
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


def _fits(x, y, r, existing_holes):
    for h in existing_holes:
        dist = ((x - h['hole_x']) ** 2 + (y - h['hole_y']) ** 2) ** 0.5
        if dist < r + h['hole_r'] + MIN_HOLE_GAP:
            return False
    return True


def sample_params(case_id):
    rng = random.Random(RANDOM_SEED + case_id)
    target_count = rng.randint(*HOLE_COUNT_RANGE)

    holes = []
    for _ in range(target_count):
        placed = False
        for _attempt in range(MAX_PLACEMENT_ATTEMPTS):
            hole_r = rng.uniform(*HOLE_R_RANGE)
            x_lo, x_hi = hole_r + HOLE_MARGIN, PLATE_W - hole_r - HOLE_MARGIN
            y_lo, y_hi = hole_r + HOLE_MARGIN, PLATE_H - hole_r - HOLE_MARGIN
            hole_x = rng.uniform(x_lo, x_hi)
            hole_y = rng.uniform(y_lo, y_hi)
            if _fits(hole_x, hole_y, hole_r, holes):
                holes.append({'hole_r': hole_r, 'hole_x': hole_x, 'hole_y': hole_y})
                placed = True
                break
        if not placed:
            log('  case %d: could not place hole %d/%d after %d attempts -- '
                'falling back to %d holes' % (
                    case_id, len(holes) + 1, target_count,
                    MAX_PLACEMENT_ATTEMPTS, len(holes)))
            break

    load = rng.uniform(*LOAD_RANGE)
    return {'holes': holes, 'load': load}


def build_and_submit(params, job_name, model_name):
    # Each case gets its own uniquely-named model rather than reusing/
    # deleting one shared name -- avoids Repository rename/delete edge cases
    # across the loop. For a batch of a few dozen cases the extra models
    # sitting in mdb cost negligible memory.
    if 'Model-1' in mdb.models.keys():
        mdb.models.changeKey(fromName='Model-1', toName=model_name)
    else:
        mdb.Model(name=model_name)
    model = mdb.models[model_name]

    holes = params['holes']
    load = params['load']

    sketch = model.ConstrainedSketch(
        name='PlateProfile', sheetSize=max(PLATE_W, PLATE_H) * 2.0)
    sketch.rectangle(point1=(0.0, 0.0), point2=(PLATE_W, PLATE_H))
    # 0 holes is just the rectangle with this loop skipped -- a valid
    # degenerate case, not a special one. Abaqus's sketch API supports
    # multiple internal loops in one profile natively; BaseShell() below
    # recognizes each one as a hole the same way it does for a single hole.
    for hole in holes:
        sketch.CircleByCenterPerimeter(
            center=(hole['hole_x'], hole['hole_y']),
            point1=(hole['hole_x'] + hole['hole_r'], hole['hole_y']))

    part = model.Part(
        name='Plate', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=sketch)

    material = model.Material(name='Steel')
    material.Elastic(table=((YOUNGS_MODULUS, POISSONS_RATIO),))
    material.Plastic(table=PLASTIC_TABLE)
    model.HomogeneousSolidSection(
        name='PlateSection', material='Steel', thickness=1.0)
    part.SectionAssignment(
        region=regionToolset.Region(faces=part.faces),
        sectionName='PlateSection')

    assembly = model.rootAssembly
    instance = assembly.Instance(name=INSTANCE_NAME, part=part, dependent=ON)

    model.StaticStep(name='Load', previous='Initial', nlgeom=OFF)

    bottom_edges = instance.edges.getByBoundingBox(yMin=-BB_TOL, yMax=BB_TOL)
    top_edges = instance.edges.getByBoundingBox(
        yMin=PLATE_H - BB_TOL, yMax=PLATE_H + BB_TOL)
    corner_vertices = instance.vertices.getByBoundingBox(
        xMin=-BB_TOL, xMax=BB_TOL, yMin=-BB_TOL, yMax=BB_TOL)

    bottom_set = assembly.Set(name='BOTTOM_EDGE_SET', edges=bottom_edges)
    top_set = assembly.Set(name='TOP_EDGE_SET', edges=top_edges)

    model.DisplacementBC(
        name='BottomFixed', createStepName='Initial', region=bottom_set, u2=0.0)
    model.DisplacementBC(
        name='CornerPin', createStepName='Initial',
        region=regionToolset.Region(vertices=corner_vertices), u1=0.0)
    model.DisplacementBC(
        name='TopLoad', createStepName='Load', region=top_set, u2=load)

    part.setElementType(
        regions=(part.faces,),
        elemTypes=(ElemType(elemCode=CPS4R, elemLibrary=STANDARD),))
    part.seedPart(size=MESH_SIZE)
    for hole in holes:
        hole_edges = part.edges.getByBoundingCylinder(
            center1=(hole['hole_x'], hole['hole_y'], -1.0),
            center2=(hole['hole_x'], hole['hole_y'], 1.0),
            radius=hole['hole_r'] + BB_TOL)
        part.seedEdgeBySize(
            edges=hole_edges,
            size=max(hole['hole_r'] * MESH_SIZE_HOLE_FACTOR, 0.3),
            constraint=FINER)
    part.generateMesh()
    assembly.regenerate()

    job = mdb.Job(name=job_name, model=model_name)
    job.submit()
    job.waitForCompletion()
    # job.status / mdb.jobs[name].status are unreliable in headless noGUI
    # sessions (observed staying None on runs that the .sta file confirms
    # completed successfully) -- check the .sta file's own completion line
    # instead, which is always written by Abaqus/Standard.
    return job_completed_successfully(job_name)


def job_completed_successfully(job_name):
    sta_path = job_name + '.sta'
    if not os.path.exists(sta_path):
        return False
    with open(sta_path, 'r') as f:
        contents = f.read()
    return 'COMPLETED SUCCESSFULLY' in contents


def extract_case(job_name, params, case_id):
    odb = openOdb(path=job_name + '.odb')
    # Abaqus stores instance names uppercased in the odb regardless of the
    # case used when creating them in CAE (verified: 'Plate-1' -> 'PLATE-1').
    instance = odb.rootAssembly.instances[INSTANCE_NAME.upper()]

    # Abaqus's own numeric types (coordinates, field values, etc.) aren't
    # natively JSON-serializable -- cast every leaf value to plain float/int.
    nodes = dict(
        (int(n.label), [float(c) for c in n.coordinates[:2]])
        for n in instance.nodes)
    elements = dict(
        (int(e.label), [int(c) for c in e.connectivity])
        for e in instance.elements)

    frame = odb.steps['Load'].frames[-1]
    u_field = frame.fieldOutputs['U']
    displacement = dict(
        (int(v.nodeLabel), [float(d) for d in v.data]) for v in u_field.values)

    mises_subset = frame.fieldOutputs['S'].getSubset(position=ELEMENT_NODAL)
    node_mises_sum = {}
    node_mises_count = {}
    for v in mises_subset.values:
        label = int(v.nodeLabel)
        node_mises_sum[label] = node_mises_sum.get(label, 0.0) + float(v.mises)
        node_mises_count[label] = node_mises_count.get(label, 0) + 1
    von_mises = dict(
        (n, node_mises_sum[n] / node_mises_count[n]) for n in node_mises_sum)

    odb.close()

    case_data = {
        'case_id': case_id,
        'params': params,
        'nodes': nodes,
        'elements': elements,
        'displacement': displacement,
        'von_mises': von_mises,
    }
    out_path = os.path.join(OUTPUT_DIR, 'case_%03d.json' % case_id)
    with open(out_path, 'w') as f:
        json.dump(case_data, f)
    return out_path


def cleanup_job_files(job_name):
    extensions = ['.odb', '.sta', '.msg', '.prt', '.com', '.sim', '.dat',
                  '.log', '.inp', '.res', '.mdl', '.stt', '.023', '.SMABulk']
    for ext in extensions:
        path = job_name + ext
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    case_range = range(START_CASE, START_CASE + N_CASES)
    succeeded = []
    for case_id in case_range:
        out_path = os.path.join(OUTPUT_DIR, 'case_%03d.json' % case_id)
        if os.path.exists(out_path):
            log('Case %d (id=%d): already generated, skipping' % (
                case_id - START_CASE + 1, case_id))
            succeeded.append(case_id)
            continue

        params = sample_params(case_id)
        job_name = 'PlateHole_case%03d' % case_id
        model_name = 'M%03d' % case_id
        log('Case %d/%d (id=%d): %r' % (
            case_id - START_CASE + 1, N_CASES, case_id, params))
        try:
            completed = build_and_submit(params, job_name, model_name)
            log('  job completed: %s' % completed)
            if not completed:
                log('  SKIPPED (did not complete) -- leaving job files for inspection')
                continue
            out_path = extract_case(job_name, params, case_id)
            log('  extracted to %s' % out_path)
            cleanup_job_files(job_name)
            succeeded.append(case_id)
        except Exception:
            log('  FAILED with exception -- leaving job files for inspection:')
            log(traceback.format_exc())
    log('Done. %d/%d cases succeeded.' % (len(succeeded), N_CASES))


if __name__ == '__main__':
    main()
