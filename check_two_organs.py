import warp as wp, newton, numpy as np

builder = newton.ModelBuilder()
builder.gravity = 0.0   # no gravity: organs held by surrounding tissue

# material values from literature (E converted to Lame params)
# prostate ~30 kPa, bladder ~17 kPa
organs = {
    "prostate_NEW": dict(k_mu=1.03e4, k_lambda=9.3e4),   # ~30 kPa
    "bladder_NEW":  dict(k_mu=5.9e3,  k_lambda=5.3e4),   # ~17 kPa
}

for name, mat in organs.items():
    nodes = np.load(f"/workspace/tet_nodes_{name}.npy")
    elems = np.load(f"/workspace/tet_elems_{name}.npy")
    verts = [wp.vec3(float(p[0]),float(p[1]),float(p[2])) for p in nodes]
    indices = [int(i) for i in elems.flatten()]
    builder.add_soft_mesh(
        pos=wp.vec3(0.0,0.0,0.0), rot=wp.quat_identity(), scale=1.0,
        vel=wp.vec3(0.0,0.0,0.0), vertices=verts, indices=indices,
        density=1.0e3, k_mu=mat["k_mu"], k_lambda=mat["k_lambda"], k_damp=5.0e-1,
    )
    print(f"added {name}: {len(verts)} verts, {len(indices)//4} tets")

builder.color()
model = builder.finalize()
solver = newton.solvers.SolverVBD(model=model, iterations=10,
    particle_enable_self_contact=False, particle_enable_tile_solve=False)
s0,s1 = model.state(), model.state()
control,contacts = model.control(), model.contacts()

dt=(1.0/60)/80
for frame in range(60):
    for _ in range(80):
        s0.clear_forces(); model.collide(s0,contacts)
        solver.step(s0,s1,control,contacts,dt); s0,s1=s1,s0
    if frame % 20 == 0 or frame==59:
        p = s0.particle_q.numpy()
        print(f"frame {frame:3d}  max|pos|={np.abs(p).max():8.2f}  finite={np.isfinite(p).all()}")
