import pyvista as pv
import tetgen
import numpy as np

# Load the real prostate mesh
mesh = pv.read("Prostate.stl")
mesh = mesh.clean().triangulate()
print("Surface:", mesh.n_points, "points,", mesh.n_cells, "faces")

tet = tetgen.TetGen(mesh)
tet.tetrahedralize()
nodes = tet.node
elems = tet.elem
print("Tet mesh:", len(nodes), "points,", len(elems), "tetrahedra")

# center + scale so it behaves like the earlier shapes
nodes = nodes - nodes.mean(axis=0)
scale = 1.0  # keep real-world size, do not shrink
nodes = nodes * scale
print("Scaled + centered. Max extent:", np.max(np.abs(nodes)))

np.save("/workspace/tet_nodes.npy", nodes)
np.save("/workspace/tet_elems.npy", elems)
print("Saved tet_nodes.npy and tet_elems.npy")
