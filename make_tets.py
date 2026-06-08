import pyvista as pv
from pyvista import examples
import tetgen
import numpy as np

# Load a real mesh (Stanford bunny) instead of a generated sphere
mesh = examples.download_bunny_coarse()
print("Loaded mesh:", mesh.n_points, "points,", mesh.n_cells, "faces")

# TetGen needs a clean triangle surface; this tidies it up
mesh = mesh.clean()
mesh = mesh.triangulate()

# Tetrahedralize the inside
tet = tetgen.TetGen(mesh)
tet.tetrahedralize()

nodes = tet.node
elems = tet.elem
print("Tet mesh:", len(nodes), "points,", len(elems), "tetrahedra")

# Normalize size + center it so it behaves like the sphere did
nodes = nodes - nodes.mean(axis=0)
scale = 0.5 / np.max(np.abs(nodes))
nodes = nodes * scale
print("Scaled + centered. Max extent now:", np.max(np.abs(nodes)))

np.save("/workspace/tet_nodes.npy", nodes)
np.save("/workspace/tet_elems.npy", elems)
print("Saved tet_nodes.npy and tet_elems.npy")
