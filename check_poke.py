import warp as wp, newton, numpy as np
nodes = np.load("/workspace/tet_nodes.npy"); elems = np.load("/workspace/tet_elems.npy")
builder = newton.ModelBuilder()
builder.gravity = 0.0
verts = [wp.vec3(float(p[0]),float(p[1]),float(p[2])) for p in nodes]
indices = [int(i) for i in elems.flatten()]
builder.add_soft_mesh(pos=wp.vec3(0.0,0.0,0.0), rot=wp.quat_identity(), scale=1.0,
    vel=wp.vec3(0.0,0.0,0.0), vertices=verts, indices=indices,
    density=1.0e3, k_mu=1.03e4, k_lambda=9.3e4, k_damp=5.0e-1)
builder.color(); model = builder.finalize()
solver = newton.solvers.SolverVBD(model=model, iterations=10,
    particle_enable_self_contact=False, particle_enable_tile_solve=False)
s0,s1 = model.state(), model.state(); control,contacts = model.control(), model.contacts()

start = s0.particle_q.numpy().copy()
top = nodes[:,1] > nodes[:,1].max() - 5
vel = s0.particle_qd.numpy(); vel[top,1] = -5.0
s0.particle_qd = wp.array(vel, dtype=wp.vec3)

dt=(1.0/60)/80
for frame in range(120):
    for _ in range(80):
        s0.clear_forces(); model.collide(s0,contacts)
        solver.step(s0,s1,control,contacts,dt); s0,s1=s1,s0
    if frame % 20 == 0 or frame==119:
        p = s0.particle_q.numpy()
        disp = p - start
        max_move = np.linalg.norm(disp, axis=1).max()
        top_move = (p[top,1] - start[top,1]).mean()
        print(f"frame {frame:3d}  max node moved {max_move:6.2f}  top dipped {top_move:7.2f}  finite={np.isfinite(p).all()}")
