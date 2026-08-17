# -*- coding: mbcs -*-
# Ad-hoc: validate extract_case() logic against an already-solved odb.
import sys
sys.path.insert(0, r'C:\Users\revan\OneDrive\Documents\Desktop\fem-plate-gnn\src')
from generate_dataset import extract_case

out_path = extract_case('PlateHole_case000', {'hole_r': 13.13}, 0)
print('wrote %s' % out_path)
