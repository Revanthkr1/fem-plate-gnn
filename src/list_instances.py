# -*- coding: mbcs -*-
from odbAccess import openOdb
odb = openOdb(path='PlateHole_case000.odb')
print('instance names: %s' % list(odb.rootAssembly.instances.keys()))
odb.close()
