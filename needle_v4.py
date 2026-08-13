# needle_v4.py: same physics as needle_v3, demo-tuned for VISIBLE indentation.
#
# Changes vs v3, and why:
# 1. NEEDLE_SPEED 6 -> 10 mm/s and 120 -> 150 frames. In your v3 run the needle
#    spent 105 of 120 frames just crossing the 2mm gap plus its own half-extent
#    travel, and only penetrated ~1.4mm. Now it reaches tissue early and drives
#    ~23mm in, a realistic biopsy-scale insertion depth.
# 2. density 1e3 -> 1e2. DEMO KNOB, flagged honestly: in this mm-unit world,
#    density 1e3 makes node masses so large the contact barely moves them.
#    Lowering density makes the tissue respond visibly. This is part of the
#    unit debt (step 4), not a calibrated value. k_mu / k_lambda stay at the
#    literature-derived numbers.
# 3. Added penetration-depth print so you can see how deep the tip is.
#
# If finite=False appears: first put density back to 1e3, rerun; if still bad,
# drop CONTACT_KE to 1e8. Paste whatever you get.

import warp as wp, newton, numpy as np, pyvista as pv, tetgen

CONTACT_KE = 1.0e9
CONTACT_KD = 1.0e-6
CONTACT_KF = 1.0e3
CONTACT_MU = 0.3
NEEDLE_SPEED = 10.0   # mm/s downward
FRAMES = 150

m = pv.read("Prostate.stl").clean().triangulate()
t = tetgen.TetGen(m); t.tetrahedralize()
nodes = t.node
verts = [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in nodes]
indices = [int(i) for i in t.elem.flatten()]

builder = newton.ModelBuilder()
builder.gravity = 0.0
builder.particle_max_velocity = 1.0e3
builder.add_soft_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1.0,
    vel=wp.vec3(0, 0, 0), vertices=verts, indices=indices,
    density=1e2, k_mu=1.03e4, k_lambda=9.3e4, k_damp=5e-1)
print(f"prostate: {len(verts)} nodes")

top = nodes.max(axis=0); ctr = nodes.mean(axis=0)
top_z0 = float(top[2])
needle_cfg = newton.ModelBuilder.ShapeConfig(
    density=1000.0, ke=CONTACT_KE, kd=CONTACT_KD, kf=CONTACT_KF, mu=CONTACT_MU)
needle_start = wp.vec3(float(ctr[0]), float(ctr[1]), top_z0 + 17.0)
b = builder.add_body(xform=wp.transform(needle_start, wp.quat_identity()))
builder.add_shape_capsule(body=b, radius=3.0, half_height=12.0, cfg=needle_cfg)
print("needle tip 2mm above prostate, center at", needle_start)

builder.color()
model = builder.finalize()
model.soft_contact_ke = CONTACT_KE
model.soft_contact_kd = CONTACT_KD
model.soft_contact_kf = CONTACT_KF
model.soft_contact_mu = CONTACT_MU

solver = newton.solvers.SolverVBD(model=model, iterations=10,
    particle_enable_self_contact=False, particle_enable_tile_solve=False,
    rigid_contact_hard=False, rigid_body_particle_contact_buffer_size=4096)
s0, s1 = model.state(), model.state()
control, contacts = model.control(), model.contacts()

drive = wp.array(np.array([[0, 0, -NEEDLE_SPEED, 0, 0, 0]], dtype=np.float32),
                 dtype=wp.spatial_vector)

viewer = newton.viewer.ViewerUSD(output_path="/workspace/needle_v4.usd", fps=60)
viewer.set_model(model)

p_rest = s0.particle_q.numpy().copy()
top_mask = p_rest[:, 2] > top_z0 - 5.0
dt = (1.0 / 60) / 80
nc = 0
for frame in range(FRAMES):
    for sub in range(80):
        wp.copy(s0.body_qd, drive)
        s0.clear_forces()
        model.collide(s0, contacts)
        if sub == 0:
            nc = int(contacts.soft_contact_count.numpy()[0])
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
    viewer.begin_frame(frame / 60.0); viewer.log_state(s0); viewer.end_frame()
    if frame % 15 == 0 or frame == FRAMES - 1:
        p = s0.particle_q.numpy()
        bq = s0.body_q.numpy()
        tip_z = bq[0, 2] - 15.0
        depth = max(0.0, top_z0 - tip_z)
        dent = float((p_rest[top_mask, 2] - p[top_mask, 2]).max())
        print(f"frame {frame:3d}  needle_z={bq[0,2]:8.2f}  tip_depth={depth:6.2f}mm  "
              f"contacts={nc:5d}  max_dent={dent:6.2f}mm  finite={np.isfinite(p).all()}")
viewer.close()
print("Saved /workspace/needle_v4.usd")
