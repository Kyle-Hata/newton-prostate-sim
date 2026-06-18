import warp as wp, newton, numpy as np

builder = newton.ModelBuilder()
builder.gravity = 0.0

organs = {
    "prostate_NEW": dict(k_mu=1.03e4, k_lambda=9.3e4),   # ~30 kPa
    "bladder_NEW":  dict(k_mu=5.9e3,  k_lambda=5.3e4),   # ~17 kPa
}

# remember where each organ's particles start in the global particle array
ranges = {}
start = 0
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
    ranges[name] = (start, start+len(verts))
    start += len(verts)
    print(f"added {name}: verts {ranges[name]}")

builder.color()
model = builder.finalize()

# poke: give the prostate's top nodes a downward velocity kick
qs = model.particle_q.numpy()
vel = model.particle_qd.numpy()
p0,p1 = ranges["prostate_NEW"]
pros = qs[p0:p1]
top_y = pros[:,1].max()
top_nodes = np.where(pros[:,1] > top_y - 5.0)[0] + p0   # within 5mm of top
vel[top_nodes,1] = -5.0
model.particle_qd = wp.array(vel, dtype=wp.vec3)
print(f"poking {len(top_nodes)} prostate top-nodes downward")

solver = newton.solvers.SolverVBD(model=model, iterations=10,
    particle_enable_self_contact=False, particle_enable_tile_solve=False)
s0,s1 = model.state(), model.state()
control,contacts = model.control(), model.contacts()

viewer = newton.viewer.ViewerUSD(output_path="/workspace/poke_two_organs.usd", fps=60)
viewer.set_model(model)

dt=(1.0/60)/80
for frame in range(120):
    for _ in range(80):
        s0.clear_forces(); model.collide(s0,contacts)
        solver.step(s0,s1,control,contacts,dt); s0,s1=s1,s0
    viewer.begin_frame(frame/60.0); viewer.log_state(s0); viewer.end_frame()
    if frame % 30 == 0 or frame==119:
        p = s0.particle_q.numpy()
        print(f"frame {frame:3d}  max|pos|={np.abs(p).max():8.2f}  finite={np.isfinite(p).all()}")

viewer.close()
print("Saved /workspace/poke_two_organs.usd")
