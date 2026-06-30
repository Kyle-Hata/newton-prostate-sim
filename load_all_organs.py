import warp as wp, newton, numpy as np, pyvista as pv, tetgen

soft = {
    "Prostate": dict(k_mu=1.03e4, k_lambda=9.3e4),
    "Bladder":  dict(k_mu=5.9e3,  k_lambda=5.3e4),
    "Rectum":   dict(k_mu=8.0e2,  k_lambda=7.0e3),
    "Urethra":  dict(k_mu=1.03e4, k_lambda=9.3e4),
}
rigid = ["Bone_1", "Bone_2"]

builder = newton.ModelBuilder()
builder.gravity = 0.0

for name, mat in soft.items():
    m = pv.read(f"{name}.stl").clean().triangulate()
    t = tetgen.TetGen(m); t.tetrahedralize()
    nodes, elems = t.node, t.elem
    verts = [wp.vec3(float(p[0]),float(p[1]),float(p[2])) for p in nodes]
    indices = [int(i) for i in elems.flatten()]
    builder.add_soft_mesh(pos=wp.vec3(0,0,0), rot=wp.quat_identity(), scale=1.0,
        vel=wp.vec3(0,0,0), vertices=verts, indices=indices,
        density=1e3, k_mu=mat["k_mu"], k_lambda=mat["k_lambda"], k_damp=5e-1)
    print(f"soft: {name} ({len(verts)} nodes)")

builder.color()
model = builder.finalize()
solver = newton.solvers.SolverVBD(model=model, iterations=5,
    particle_enable_self_contact=False, particle_enable_tile_solve=False)
s0,s1 = model.state(), model.state()
control,contacts = model.control(), model.contacts()

viewer = newton.viewer.ViewerUSD(output_path="/workspace/all_organs.usd", fps=60)
viewer.set_model(model)

dt=(1.0/60)/40
for frame in range(60):
    for _ in range(40):
        s0.clear_forces(); model.collide(s0,contacts)
        solver.step(s0,s1,control,contacts,dt); s0,s1=s1,s0
    viewer.begin_frame(frame/60.0); viewer.log_state(s0); viewer.end_frame()
    if frame%20==0 or frame==59:
        p=s0.particle_q.numpy()
        print(f"frame {frame}  max|pos|={np.abs(p).max():.1f}  finite={np.isfinite(p).all()}")
viewer.close()
print("Saved /workspace/all_organs.usd")
