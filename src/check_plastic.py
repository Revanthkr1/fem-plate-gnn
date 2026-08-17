from odbAccess import openOdb
odb = openOdb(path='PlateHole_Plastic_v1.odb')
frame = odb.steps['Load'].frames[-1]
mises_vals = [v.mises for v in frame.fieldOutputs['S'].values]
print('max mises: %.4f' % max(mises_vals))
try:
    peeq_vals = [v.data for v in frame.fieldOutputs['PEEQ'].values]
    print('max PEEQ: %.6f' % max(peeq_vals))
    print('num integration points with PEEQ > 0: %d / %d' % (sum(1 for p in peeq_vals if p > 1e-8), len(peeq_vals)))
except KeyError:
    print('No PEEQ field output found')
odb.close()
