# full_scene_xpbd.py: ALL organs + bones + needle, FEM soft bodies, SolverXPBD.
#
# Why this is the honest favorite: XPBD handles rigid-soft contact as a
# positional projection (per the comment in Newton's own
# example_rigid_soft_contact.py) so no 1e9 stiffness band-aid, and it has
# particle-particle contact built into the solver (solve_particle_particle_
# contacts in newton/_src/solvers/xpbd/solver_xpbd.py), which is the
# organ-organ coupling. This file uses REAL density 1e3 and ordinary contact
# numbers. Unit debt still applies to the elastic side.
#
# For a first run set FRAMES = 30 to check speed, then 150.

import warp as wp, newton, numpy as np, pyvista as pv, tetgen

@wp.kernel
def damp_vel(qd: wp.array(dtype=wp.vec3), f: float):
    i = wp.tid()
    qd[i] = qd[i] * f


NEEDLE_SPEED = 10.0
SETTLE_FRAMES = 20
FRAMES = 150
OUT = "/workspace/full_scene_xpbd.usd"

SOFT = {
    "Prostate": dict(k_mu=1.03e4, k_lambda=9.3e4),
    "Bladder":  dict(k_mu=5.9e3,  k_lambda=5.3e4),
    "Rectum":   dict(k_mu=8.0e2,  k_lambda=7.0e3),
    "Urethra":  dict(k_mu=1.03e4, k_lambda=9.3e4),
}
STATIC = ["Bone_1", "Bone_2", "Diaphragm"]  # Diaphragm = pelvic-floor sheet here; static context

def read_organ(name):
    # prefer the Segmentation_ export set so every organ shares one frame;
    # the repo's standalone Prostate.stl is a different, older export
    for fn in (f"Segmentation_of_prostate_{name}.stl", f"Segmentation of prostate_{name}.stl", f"{name}.stl"):
        try:
            return pv.read(fn).clean().triangulate()
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"no STL for {name}")

surfs = {}
for name in SOFT:
    surfs[name] = read_organ(name)

pro = surfs["Prostate"]
ctr = np.array(pro.center)
top_z, bot_z = float(pro.bounds[5]), float(pro.bounds[4])
others = [n for n in SOFT if n != "Prostate"]
def path_crossings(start, end):
    return [n for n in others if len(surfs[n].ray_trace(start, end)[0]) > 0]
best = None
for dx in (0.0, -6.0, 6.0, -10.0, 10.0):
    for dy in (0.0, 6.0, -6.0, 10.0, -10.0):
        for label, z0, sgn in (("transperineal ascending", bot_z - 17.0, +1.0),
                               ("superior descending", top_z + 17.0, -1.0)):
            start = np.array([ctr[0] + dx, ctr[1] + dy, z0])
            end = np.array([start[0], start[1], ctr[2]])
            if len(pro.ray_trace(start, end)[0]) == 0:
                continue
            blk = path_crossings(start, end)
            score = (len(blk), abs(dx) + abs(dy), 0 if sgn > 0 else 1)
            if best is None or score < best[0]:
                best = (score, label, start, sgn, blk)
score, label, needle_start_np, sgn, blk = best
print(f"chosen approach: {label} at offset "
      f"({needle_start_np[0]-ctr[0]:+.0f},{needle_start_np[1]-ctr[1]:+.0f}) "
      f"crosses {blk if blk else 'nothing'}")

builder = newton.ModelBuilder()
builder.gravity = 0.0
builder.particle_max_velocity = 1.0e3
builder.default_particle_radius = 0.5

ranges = {}
for name, mat in SOFT.items():
    t = tetgen.TetGen(surfs[name]); t.tetrahedralize()
    verts = [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in t.node]
    indices = [int(i) for i in t.elem.flatten()]
    i0 = builder.particle_count
    builder.add_soft_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1.0,
        vel=wp.vec3(0, 0, 0), vertices=verts, indices=indices,
        density=1e3, k_mu=mat["k_mu"], k_lambda=mat["k_lambda"], k_damp=5e-1)
    ranges[name] = (i0, builder.particle_count)
    print(f"soft: {name} ({builder.particle_count - i0} nodes)")

shape_cfg = newton.ModelBuilder.ShapeConfig(
    density=1000.0, ke=1.0e3, kd=10.0, kf=1.0e3, mu=0.5)
for name in STATIC:
    try:
        bm = read_organ(name)
    except FileNotFoundError:
        print(f"SKIPPED missing {name}.stl")
        continue
    faces = bm.faces.reshape(-1, 4)[:, 1:].astype(np.int32).flatten()
    mesh = newton.Mesh(np.asarray(bm.points, dtype=np.float32), faces)
    builder.add_shape_mesh(body=-1, mesh=mesh, cfg=shape_cfg)
    print(f"rigid static: {name} ({bm.n_points} verts)")

b = builder.add_body(xform=wp.transform(
    wp.vec3(float(needle_start_np[0]), float(needle_start_np[1]), float(needle_start_np[2])),
    wp.quat_identity()))
builder.add_shape_capsule(body=b, radius=3.0, half_height=12.0, cfg=shape_cfg)

model = builder.finalize()
model.soft_contact_ke = 1.0e3
model.soft_contact_kd = 10.0
model.soft_contact_kf = 1.0e3
model.soft_contact_mu = 0.5

solver = newton.solvers.SolverXPBD(model=model, iterations=10)
s0, s1 = model.state(), model.state()
control, contacts = model.control(), model.contacts()

drive = wp.array(np.array([[0, 0, sgn * NEEDLE_SPEED, 0, 0, 0]], dtype=np.float32),
                 dtype=wp.spatial_vector)

dt = (1.0 / 60) / 80
zero_drive = wp.zeros(1, dtype=wp.spatial_vector)
p_raw = s0.particle_q.numpy().copy()
for frame in range(SETTLE_FRAMES):
    for _ in range(80):
        wp.copy(s0.body_qd, zero_drive)
        s0.clear_forces()
        model.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
        wp.launch(damp_vel, dim=model.particle_count, inputs=[s0.particle_qd, 0.2])
    if frame % 5 == 0 or frame == SETTLE_FRAMES - 1:
        p = s0.particle_q.numpy()
        print(f"settle {frame:2d}  drift={np.linalg.norm(p - p_raw, axis=1).max():5.2f}mm  "
              f"finite={np.isfinite(p).all()}")
# stage 2: light damping until residual overlap stops moving the scene
prev = s0.particle_q.numpy().copy()
for frame in range(60):
    for _ in range(80):
        wp.copy(s0.body_qd, zero_drive)
        s0.clear_forces()
        model.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
        wp.launch(damp_vel, dim=model.particle_count, inputs=[s0.particle_qd, 0.9])
    p = s0.particle_q.numpy()
    step_move = float(np.linalg.norm(p - prev, axis=1).max())
    prev = p.copy()
    ncr = int(contacts.soft_contact_count.numpy()[0])
    if frame % 5 == 0:
        print(f"relax  {frame:2d}  per-frame move={step_move:5.3f}mm  contacts={ncr:5d}  "
              f"finite={np.isfinite(p).all()}")
    if step_move < 0.05:
        print(f"relaxed after {frame + 1} extra frames (per-frame move {step_move:.3f}mm)")
        break
wp.launch(damp_vel, dim=model.particle_count, inputs=[s0.particle_qd, 0.0])

viewer = newton.viewer.ViewerUSD(output_path=OUT, fps=60)
viewer.set_model(model)

p_rest = s0.particle_q.numpy().copy()
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
        tip_z = bq[0, 2] + sgn * 15.0
        depth = max(0.0, (top_z - tip_z) if sgn < 0 else (tip_z - bot_z))
        per = "  ".join(
            f"{n}={np.linalg.norm(p[a:z2] - p_rest[a:z2], axis=1).max():5.2f}mm"
            for n, (a, z2) in ranges.items())
        print(f"frame {frame:3d}  tip_depth={depth:6.2f}mm  contacts={nc:5d}  "
              f"{per}  finite={np.isfinite(p).all()}")
viewer.close()
print(f"Saved {OUT}")
