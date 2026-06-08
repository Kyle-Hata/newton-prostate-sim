import pyvista as pv
import tetgen
import numpy as np

# A torus: watertight but has a hole through it (Euler number 0)
mesh = pv.ParametricTorus(ringradius=1.0, crosssectionradius=0.3)
mesh = mesh.clean().triangulate()
print("Surface:", mesh.n_points, "points,", mesh.n_cells, "faces")

tet = tetgen.TetGen(mesh)
tet.tetrahedralize()
nodes = tet.node
elems = tet.elem
print("Tet mesh:", len(nodes), "points,", len(elems), "tetrahedra")

# center + scale to behave like before
nodes = nodes - nodes.mean(axis=0)
scale = 0.5 / np.max(np.abs(nodes))
nodes = nodes * scale
print("Scaled. Max extent:", np.max(np.abs(nodes)))

np.save("/workspace/tet_nodes.npy", nodes)
np.save("/workspace/tet_elems.npy", elems)
print("Saved.")
