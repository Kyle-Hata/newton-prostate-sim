import pyvista as pv
import tetgen
import numpy as np

# Make a clean sphere surface mesh (guaranteed watertight)
sphere = pv.Sphere(radius=0.3)
print("Surface mesh:", sphere.n_points, "points,", sphere.n_cells, "triangles")

# Tetrahedralize the inside
tet = tetgen.TetGen(sphere)
tet.tetrahedralize()

# Pull the results off the object
nodes = tet.node          # vertex positions
elems = tet.elem          # tetrahedra (4 indices each)

print("Tet mesh:", len(nodes), "points,", len(elems), "tetrahedra")

# Save the tet data so the next script can use it
np.save("/workspace/tet_nodes.npy", nodes)
np.save("/workspace/tet_elems.npy", elems)
print("Saved tet_nodes.npy and tet_elems.npy")
