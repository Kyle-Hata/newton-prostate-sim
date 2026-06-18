import warp as wp, newton, numpy as np

builder = newton.ModelBuilder()
builder.gravity = 0.0

organs = {
    "prostate_NEW": dict(k_mu=1.03e4, k_lambda=9.3e4),   # ~30 kPa
    "bladder_NEW":  dict(k_mu=5.9e3,  k_lambda=5.3e4),   # ~17 kPa
}

ranges = {}; start = 0
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
    ranges[name] = (start, start+len(verts)); start += len(verts)
    print(f"added {name}: verts {ranges[name]}")

builder.color()
model = builder.finalize()

# figure out which axis points from prostate -> bladder, push along it
pq = model.particle_q.numpy()
p0,p1 = ranges["prostate_NEW"]; b0,b1 = ranges["bladder_NEW"]
pros_c = pq[p0:p1].mean(axis=0); blad_c = pq[b0:b1].mean(axis=0)
direction = blad_c - pros_c
direction = direction / np.linalg.norm(direction)
print("push direction (prostate->bladder):", np.round(direction,2))

# give the WHOLE prostate a velocity toward the bladder
vel = model.particle_qd.numpy()
push_speed = 30.0   # tune this; higher = harder push
vel[p0:p1] = direction * push_speed
model.particle_qd = wp.array(vel, dtype=wp.vec3)
print(f"pushing all {p1-p0} prostate nodes toward bladder at speed {push_speed}")

# contact ON: let particles of different organs collide
solver = newton.solvers.SolverVBD(model=model, iterations=10,
    particle_enable_self_contact=True, particle_enable_tile_solve=False)
s0,s1 = model.state(), model.state()
control,contacts = model.control(), model.contacts()

viewer = newton.viewer.ViewerUSD(output_path="/workspace/push_into_bladder.usd", fps=60)
viewer.set_model(model)

dt=(1.0/60)/80
for frame in range(120):
    for _ in range(80):
        s0.clear_forces(); model.collide(s0,contacts)
        solver.step(s0,s1,control,contacts,dt); s0,s1=s1,s0
    viewer.begin_frame(frame/60.0); viewer.log_state(s0); viewer.end_frame()
    if frame % 30 == 0 or frame==119:
        p = s0.particle_q.numpy()
        bc = p[b0:b1].mean(axis=0)
        print(f"frame {frame:3d}  bladder_center={np.round(bc,1)}  max|pos|={np.abs(p).max():7.2f}  finite={np.isfinite(p).all()}")

viewer.close()
print("Saved /workspace/push_into_bladder.usd")
