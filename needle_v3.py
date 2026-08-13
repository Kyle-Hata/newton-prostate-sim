# needle_v3.py: rigid capsule indenting the soft prostate. Newton 1.3.0.
#
# What changed vs needle_v2.py, with sources:
# 1. body_qd is LINEAR first, ANGULAR last in Newton (newton/_src/sim/state.py,
#    State.body_qd docstring, v1.3.0). bqd[0,5] was angular-z, so the needle
#    was spinning in place and never descended. Down is bqd[0,2] = -speed.
# 2. Velocity is re-asserted every substep, so contact cannot stall the needle.
#    This is a prescribed-motion tool, the standard surgical-sim setup.
# 3. Contact stiffness set on BOTH sides. VBD body-particle contact uses the
#    average of model.soft_contact_ke and the shape's ke
#    (newton/_src/solvers/vbd/rigid_vbd_kernels.py, init_body_particle_contacts).
#    Defaults are 1e3 and 2.5e3, far too weak against mm-scale node masses.
#    Mirrors newton/examples/multiphysics/example_rigid_soft_contact.py (vbd
#    branch), which sets shape cfg ke and model.soft_contact_ke together.
# 4. VBD flags from that example: rigid_contact_hard=False and a
#    rigid_body_particle_contact_buffer_size large enough for a fine tet surface.
#
# CONTACT_KE=1e9 is deliberately huge because the scene is mm + density 1e3,
# which makes node masses ~1e4 or more. It is a band-aid until the unit pass
# (step 4). Tune it using the printed dent, not by eye.

import warp as wp, newton, numpy as np, pyvista as pv, tetgen

CONTACT_KE = 1.0e9
CONTACT_KD = 1.0e-6   # VBD damping coeff = kd * ke, so keep kd tiny when ke is huge
CONTACT_KF = 1.0e3
CONTACT_MU = 0.3
NEEDLE_SPEED = 6.0    # mm/s downward

m = pv.read("Prostate.stl").clean().triangulate()
t = tetgen.TetGen(m); t.tetrahedralize()
nodes = t.node
verts = [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in nodes]
indices = [int(i) for i in t.elem.flatten()]

builder = newton.ModelBuilder()
builder.gravity = 0.0
builder.particle_max_velocity = 1.0e3   # default clamp is meter-scale, too low for mm/s
builder.add_soft_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1.0,
    vel=wp.vec3(0, 0, 0), vertices=verts, indices=indices,
    density=1e3, k_mu=1.03e4, k_lambda=9.3e4, k_damp=5e-1)
print(f"prostate: {len(verts)} nodes")

top = nodes.max(axis=0); ctr = nodes.mean(axis=0)
# capsule extends +/-(half_height + radius) = +/-15 along its local Z axis,
# so center at top+17 puts the tip 2 mm above the surface
needle_cfg = newton.ModelBuilder.ShapeConfig(
    density=1000.0, ke=CONTACT_KE, kd=CONTACT_KD, kf=CONTACT_KF, mu=CONTACT_MU)
needle_start = wp.vec3(float(ctr[0]), float(ctr[1]), float(top[2] + 17.0))
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

# constant downward drive: [vx, vy, vz, wx, wy, wz], linear first
drive = wp.array(np.array([[0, 0, -NEEDLE_SPEED, 0, 0, 0]], dtype=np.float32),
                 dtype=wp.spatial_vector)

viewer = newton.viewer.ViewerUSD(output_path="/workspace/needle_v3.usd", fps=60)
viewer.set_model(model)

p_rest = s0.particle_q.numpy().copy()
top_mask = p_rest[:, 2] > top[2] - 5.0
dt = (1.0 / 60) / 80
nc = 0
for frame in range(120):
    for sub in range(80):
        wp.copy(s0.body_qd, drive)
        s0.clear_forces()
        model.collide(s0, contacts)
        if sub == 0:
            nc = int(contacts.soft_contact_count.numpy()[0])
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
    viewer.begin_frame(frame / 60.0); viewer.log_state(s0); viewer.end_frame()
    if frame % 15 == 0 or frame == 119:
        p = s0.particle_q.numpy()
        bq = s0.body_q.numpy()
        dent = float((p_rest[top_mask, 2] - p[top_mask, 2]).max())
        print(f"frame {frame:3d}  needle_z={bq[0,2]:8.2f}  body_particle_contacts={nc:5d}  "
              f"max_dent={dent:6.2f}mm  finite={np.isfinite(p).all()}")
viewer.close()
print("Saved /workspace/needle_v3.usd")
