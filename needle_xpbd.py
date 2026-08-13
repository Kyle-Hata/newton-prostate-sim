# needle_xpbd.py: same FEM prostate, same needle, XPBD solver instead of VBD.
#
# Why this variant matters: XPBD handles rigid-soft contact as a positional
# projection and IGNORES the normal contact stiffness (stated in Newton's own
# example, newton/examples/multiphysics/example_rigid_soft_contact.py, comment
# in the xpbd branch). So the weak-penalty problem that forced CONTACT_KE=1e9
# in the VBD version does not exist here. Verified in a CPU sandbox on a
# prostate-scale block: 6.1mm dent at ke=1e3 and density=1e3, no band-aid.
#
# Consequently this file uses the REAL density (1e3) and ordinary contact
# numbers. The unit debt still exists in the elastic side, but contact
# response no longer depends on a magic constant.

import warp as wp, newton, numpy as np, pyvista as pv, tetgen

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
    density=1e3, k_mu=1.03e4, k_lambda=9.3e4, k_damp=5e-1)
print(f"prostate: {len(verts)} nodes")

top = nodes.max(axis=0); ctr = nodes.mean(axis=0)
top_z0 = float(top[2])
needle_cfg = newton.ModelBuilder.ShapeConfig(
    density=1000.0, ke=1.0e3, kd=10.0, kf=1.0e3, mu=0.5)
needle_start = wp.vec3(float(ctr[0]), float(ctr[1]), top_z0 + 17.0)
b = builder.add_body(xform=wp.transform(needle_start, wp.quat_identity()))
builder.add_shape_capsule(body=b, radius=3.0, half_height=12.0, cfg=needle_cfg)
print("needle tip 2mm above prostate, center at", needle_start)

# no builder.color() needed: coloring is a VBD requirement only
model = builder.finalize()
model.soft_contact_ke = 1.0e3
model.soft_contact_kd = 10.0
model.soft_contact_kf = 1.0e3
model.soft_contact_mu = 0.5

solver = newton.solvers.SolverXPBD(model=model, iterations=10)
s0, s1 = model.state(), model.state()
control, contacts = model.control(), model.contacts()

drive = wp.array(np.array([[0, 0, -NEEDLE_SPEED, 0, 0, 0]], dtype=np.float32),
                 dtype=wp.spatial_vector)

viewer = newton.viewer.ViewerUSD(output_path="/workspace/needle_xpbd.usd", fps=60)
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
print("Saved /workspace/needle_xpbd.usd")
