import numpy as np
nodes = np.load("/workspace/tet_nodes.npy")
elems = np.load("/workspace/tet_elems.npy")
print("Nodes:", nodes.shape, " Elems:", elems.shape)

# volume of each tet
v0 = nodes[elems[:,0]]
v1 = nodes[elems[:,1]]
v2 = nodes[elems[:,2]]
v3 = nodes[elems[:,3]]
vol = np.einsum('ij,ij->i', np.cross(v1-v0, v2-v0), v3-v0) / 6.0
av = np.abs(vol)
print("Tet volume  min:", av.min(), " max:", av.max(), " median:", np.median(av))
print("Tiny tets (<1e-9):", int((av < 1e-9).sum()))
print("Near-zero (<1e-12):", int((av < 1e-12).sum()))
print("Negative-volume (inverted):", int((vol < 0).sum()))
