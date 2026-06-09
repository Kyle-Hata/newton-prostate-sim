import warp as wp
import newton
import numpy as np

nodes = np.load("/workspace/tet_nodes.npy")
elems = np.load("/workspace/tet_elems.npy")
print("loaded nodes max|coord|:", np.abs(nodes).max())

builder = newton.ModelBuilder()
builder.add_ground_plane()
verts = [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in nodes]
indices = [int(i) for i in elems.flatten()]
builder.add_soft_mesh(
    pos=wp.vec3(0.0, 55.0, 0.0), rot=wp.quat_identity(), scale=1.0,
    vel=wp.vec3(0.0,0.0,0.0), vertices=verts, indices=indices,
    density=1.0e3, k_mu=5.0e4, k_lambda=5.0e4, k_damp=5.0e-1,
)
builder.color()
model = builder.finalize()
model.soft_contact_ke = 1.0e4
model.soft_contact_kd = 1.0e2
model.soft_contact_mu = 1.0
solver = newton.solvers.SolverVBD(model=model, iterations=10,
    particle_enable_self_contact=False, particle_enable_tile_solve=False)
s0, s1 = model.state(), model.state()
control, contacts = model.control(), model.contacts()
dt = (1.0/60)/80
for step in range(10):
    pos = s0.particle_q.numpy()
    print(f"step {step:2d}  max|pos|={np.abs(pos).max():.3e}  mean|pos|={np.abs(pos).mean():.3e}")
    if not np.isfinite(pos).all():
        print("  -> NON-FINITE detected"); break
    s0.clear_forces()
    model.collide(s0, contacts)
    solver.step(s0, s1, control, contacts, dt)
    s0, s1 = s1, s0
