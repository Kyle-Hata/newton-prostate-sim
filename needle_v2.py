import warp as wp, newton, numpy as np, pyvista as pv, tetgen

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

top = nodes.max(axis=0); ctr = nodes.mean(axis=0)
needle_start = wp.vec3(float(ctr[0]), float(ctr[1]), float(top[2]+8.0))
b = builder.add_body(xform=wp.transform(needle_start, wp.quat_identity()))
builder.add_shape_capsule(body=b, radius=3.0, half_height=12.0)  # fatter needle = easier contact
print("needle at", needle_start)

builder.color()
model = builder.finalize()
# bigger contact margins so soft-rigid contact actually catches (mm-scale lesson)
solver = newton.solvers.SolverVBD(model=model, iterations=10,
    particle_enable_self_contact=True,
    particle_self_contact_radius=3.0, particle_self_contact_margin=5.0,
    particle_enable_tile_solve=False)
s0,s1 = model.state(), model.state()
control,contacts = model.control(), model.contacts()

bqd = s0.body_qd.numpy()
bqd[0,5] = -6.0   # SLOWER needle so it can't tunnel through
s0.body_qd = wp.array(bqd, dtype=wp.spatial_vector)
print("driving needle down slowly")

viewer = newton.viewer.ViewerUSD(output_path="/workspace/needle_v2.usd", fps=60)
viewer.set_model(model)
top_nodes_before = nodes[nodes[:,2] > top[2]-5.0].mean(axis=0)
dt=(1.0/60)/80
for frame in range(120):
    for _ in range(80):
        s0.clear_forces(); model.collide(s0,contacts)
        solver.step(s0,s1,control,contacts,dt); s0,s1=s1,s0
    if frame%30==0 or frame==119:
        p=s0.particle_q.numpy()
        tn = p[nodes[:,2] > top[2]-5.0].mean(axis=0)
        dent = top_nodes_before[2] - tn[2]
        print(f"frame {frame}  top dented {dent:.2f}mm  finite={np.isfinite(p).all()}")
    viewer.begin_frame(frame/60.0); viewer.log_state(s0); viewer.end_frame()
viewer.close()
print("Saved /workspace/needle_v2.usd")
