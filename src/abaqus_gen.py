# -*- coding: mbcs -*-
# Abaqus/CAE scripting-interface script (Abaqus 2023 kernel = Python 2.7).
# Run with: abaqus cae noGUI=src/abaqus_gen.py
#
# Phase 1 (see plan): a single hardcoded elastic plate-with-hole case, used to
# verify the Abaqus setup against the analytical Kirsch stress-concentration
# solution (SCF ~= 3 for a small circular hole in a plate under uniaxial
# tension) before any parametrization or plasticity is added.
#
# caeModules must be star-imported (not just abaqus/abaqusConstants) -- Abaqus
# only binds methods like Model.StaticStep, Part.generateMesh, etc. onto the
# kernel objects once the corresponding feature module (step, mesh, part, ...)
# has been imported.

from abaqus import mdb
from abaqusConstants import (
    TWO_D_PLANAR, DEFORMABLE_BODY, OFF, ON, STANDARD, CPS4R, FINER,
)
from caeModules import *
import regionToolset
from mesh import ElemType

PLATE_W = 100.0
PLATE_H = 200.0
HOLE_R = 5.0
HOLE_X = PLATE_W / 2.0
HOLE_Y = PLATE_H / 2.0

YOUNGS_MODULUS = 210000.0
POISSONS_RATIO = 0.3
# Isotropic hardening table: (yield stress, plastic strain) pairs. First point
# is the initial yield stress; later points define the hardening curve.
PLASTIC_TABLE = ((250.0, 0.0), (300.0, 0.01), (400.0, 0.05))

# Elastic-only run peaked at ~143 MPa local stress for a 0.05mm top
# displacement (SCF ~2.75, verified against Kirsch). Scaling up to 0.15mm
# pushes the elastic-equivalent peak to ~430 MPa, comfortably past the 250 MPa
# yield point so the hole edge actually yields and redistributes load.
TOP_DISPLACEMENT = 0.15

MESH_SIZE = 2.0
MESH_SIZE_HOLE = 0.5

MODEL_NAME = 'PlateHole'
JOB_NAME = 'PlateHole_Plastic_v1'

BB_TOL = 1e-3


def build_model():
    mdb.models.changeKey(fromName='Model-1', toName=MODEL_NAME)
    model = mdb.models[MODEL_NAME]

    sketch = model.ConstrainedSketch(
        name='PlateProfile', sheetSize=max(PLATE_W, PLATE_H) * 2.0)
    sketch.rectangle(point1=(0.0, 0.0), point2=(PLATE_W, PLATE_H))
    sketch.CircleByCenterPerimeter(
        center=(HOLE_X, HOLE_Y), point1=(HOLE_X + HOLE_R, HOLE_Y))

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
    instance = assembly.Instance(name='Plate-1', part=part, dependent=ON)

    model.StaticStep(name='Load', previous='Initial', nlgeom=OFF)

    bottom_edges = instance.edges.getByBoundingBox(
        yMin=-BB_TOL, yMax=BB_TOL)
    top_edges = instance.edges.getByBoundingBox(
        yMin=PLATE_H - BB_TOL, yMax=PLATE_H + BB_TOL)
    corner_vertices = instance.vertices.getByBoundingBox(
        xMin=-BB_TOL, xMax=BB_TOL, yMin=-BB_TOL, yMax=BB_TOL)

    # Named sets so the reaction-force sum (used for the Kirsch SCF check) can
    # be computed from the odb by set name, without re-deriving geometry there.
    bottom_set = assembly.Set(name='BOTTOM_EDGE_SET', edges=bottom_edges)
    assembly.Set(name='TOP_EDGE_SET', edges=top_edges)

    model.DisplacementBC(
        name='BottomFixed', createStepName='Initial',
        region=bottom_set, u2=0.0)
    model.DisplacementBC(
        name='CornerPin', createStepName='Initial',
        region=regionToolset.Region(vertices=corner_vertices), u1=0.0)
    model.DisplacementBC(
        name='TopLoad', createStepName='Load',
        region=assembly.sets['TOP_EDGE_SET'], u2=TOP_DISPLACEMENT)

    part.setElementType(
        regions=(part.faces,),
        elemTypes=(ElemType(elemCode=CPS4R, elemLibrary=STANDARD),))
    part.seedPart(size=MESH_SIZE)

    hole_edges = part.edges.getByBoundingCylinder(
        center1=(HOLE_X, HOLE_Y, -1.0), center2=(HOLE_X, HOLE_Y, 1.0),
        radius=HOLE_R + BB_TOL)
    part.seedEdgeBySize(edges=hole_edges, size=MESH_SIZE_HOLE, constraint=FINER)
    part.generateMesh()
    assembly.regenerate()

    return model


def submit_job(model):
    job = mdb.Job(name=JOB_NAME, model=MODEL_NAME)
    job.submit()
    job.waitForCompletion()
    print('Job %s status: %s' % (JOB_NAME, job.status))
    return job


if __name__ == '__main__':
    m = build_model()
    submit_job(m)
