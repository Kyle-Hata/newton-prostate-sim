import pyvista as pv
import tetgen
import numpy as np
import sys

name = sys.argv[1]   # e.g. "prostate_NEW" or "bladder_NEW"
infile = f"{name}.stl"

mesh = pv.read(infile)
mesh = mesh.clean().triangulate()
print(f"{name} surface:", mesh.n_points, "points,", mesh.n_cells, "faces")

tet = tetgen.TetGen(mesh)
tet.tetrahedralize()
nodes = tet.node
elems = tet.elem
print(f"{name} tets:", len(nodes), "points,", len(elems), "tetrahedra")

# Keep real-world mm coordinates (NO centering, NO scaling) so organs
# stay in their true anatomical positions relative to each other.
np.save(f"/workspace/tet_nodes_{name}.npy", nodes)
np.save(f"/workspace/tet_elems_{name}.npy", elems)
print(f"Saved tet_nodes_{name}.npy and tet_elems_{name}.npy")
print("  center:", np.round(nodes.mean(axis=0),1), " extent:", np.round(np.abs(nodes - nodes.mean(axis=0)).max(),1))
