import warp as wp, newton, numpy as np, pyvista as pv, tetgen

# soft prostate only (keep it fast and focused)
m = pv.read("Prostate.stl").clean().triangulate()
t = tetgen.TetGen(m); t.tetrahedralize()
nodes = t.node
verts = [wp.vec3(float(p[0]),float(p[1]),float(p[2])) for p in nodes]
indices = [int(i) for i in t.elem.flatten()]

builder = newton.ModelBuilder()
builder.gravity = 0.0
builder.add_soft_mesh(pos=wp.vec3(0,0,0), rot=wp.quat_identity(), scale=1.0,
    vel=wp.vec3(0,0,0), vertices=verts, indices=indices,
    density=1e3, k_mu=1.03e4, k_lambda=9.3e4, k_damp=5e-1)
print(f"prostate: {len(verts)} nodes")

# needle: a thin rigid capsule positioned just above the prostate's top
top = nodes.max(axis=0)
ctr = nodes.mean(axis=0)
needle_start = wp.vec3(float(ctr[0]), float(ctr[1]), float(top[2]+15.0))
b = builder.add_body(xform=wp.transform(needle_start, wp.quat_identity()))
builder.add_shape_capsule(body=b, radius=1.5, half_height=12.0)
print("needle added above prostate at", needle_start)

builder.color()
model = builder.finalize()
solver = newton.solvers.SolverVBD(model=model, iterations=8,
    particle_enable_self_contact=False, particle_enable_tile_solve=False)
s0,s1 = model.state(), model.state()
control,contacts = model.control(), model.contacts()

# drive the needle downward into the prostate
bq = s0.body_q.numpy()
bqd = s0.body_qd.numpy()
bqd[0,5] = -20.0   # linear velocity in z (downward), angular[0:3]+linear[3:6]
s0.body_qd = wp.array(bqd, dtype=wp.spatial_vector)
print("driving needle down at 20/s")

viewer = newton.viewer.ViewerUSD(output_path="/workspace/needle_indent.usd", fps=60)
viewer.set_model(model)
dt=(1.0/60)/60
for frame in range(120):
    for _ in range(60):
        s0.clear_forces(); model.collide(s0,contacts)
        solver.step(s0,s1,control,contacts,dt); s0,s1=s1,s0
    viewer.begin_frame(frame/60.0); viewer.log_state(s0); viewer.end_frame()
    if frame%30==0 or frame==119:
        p=s0.particle_q.numpy()
        print(f"frame {frame}  max|pos|={np.abs(p).max():.1f}  finite={np.isfinite(p).all()}")
viewer.close()
print("Saved /workspace/needle_indent.usd")
