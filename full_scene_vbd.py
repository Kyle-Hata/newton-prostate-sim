# full_scene_vbd.py: ALL organs + bones + needle, FEM soft bodies, SolverVBD.
#
# What this adds over your load_all_organs.py:
# - particle self-contact ON (your repo runs it off everywhere, which means
#   organs could never push each other; this is what enables organ-organ shift)
# - bones actually loaded as static collision meshes (your loader defined the
#   list but never used it)
# - the needle, driven along an approach chosen at runtime by ray-testing the
#   path against the bladder and rectum, so it goes transperineal from below
#   if a top-down path would spear the bladder first
# - per-organ displacement printed every 15 frames, including the bladder
#   shift number you have been chasing
#
# DEMO KNOBS, flagged honestly: density=1e2 and CONTACT_KE=1e9 are the same
# unit-debt band-aids as needle_v4. Deformation magnitudes are not calibrated.
#
# Runtime warning: 4 organs with self-contact is much heavier than one organ.
# For a first run set FRAMES = 30 to check speed and stability, then 150.

import warp as wp, newton, numpy as np, pyvista as pv, tetgen

CONTACT_KE = 1.0e9
CONTACT_KD = 1.0e-6
CONTACT_KF = 1.0e3
CONTACT_MU = 0.3
NEEDLE_SPEED = 10.0
FRAMES = 150
OUT = "/workspace/full_scene_vbd.usd"

SOFT = {
    "Prostate": dict(k_mu=1.03e4, k_lambda=9.3e4),
    "Bladder":  dict(k_mu=5.9e3,  k_lambda=5.3e4),
    "Rectum":   dict(k_mu=8.0e2,  k_lambda=7.0e3),
    "Urethra":  dict(k_mu=1.03e4, k_lambda=9.3e4),
}
BONES = ["Bone_1", "Bone_2"]

surfs = {}
for name in SOFT:
    surfs[name] = pv.read(f"{name}.stl").clean().triangulate()

# choose needle approach by ray-testing candidate paths against bladder/rectum
pro = surfs["Prostate"]
ctr = np.array(pro.center)
top_z, bot_z = float(pro.bounds[5]), float(pro.bounds[4])
candidates = {
    "superior descending":       (np.array([ctr[0], ctr[1], top_z + 17.0]), -1.0),
    "transperineal ascending":   (np.array([ctr[0], ctr[1], bot_z - 17.0]), +1.0),
}
best = None
for label, (start, sgn) in candidates.items():
    blockers = [n for n in ("Bladder", "Rectum")
                if len(surfs[n].ray_trace(start, ctr)[0]) > 0]
    print(f"approach {label}: crosses {blockers if blockers else 'nothing'}")
    if best is None or len(blockers) < best[3]:
        best = (label, start, sgn, len(blockers))
label, needle_start_np, sgn, _ = best
print(f"chosen approach: {label}")

builder = newton.ModelBuilder()
builder.gravity = 0.0
builder.particle_max_velocity = 1.0e3
builder.default_particle_radius = 1.0  # contact thickness for organ-organ

ranges = {}
for name, mat in SOFT.items():
    t = tetgen.TetGen(surfs[name]); t.tetrahedralize()
    verts = [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in t.node]
    indices = [int(i) for i in t.elem.flatten()]
    i0 = builder.particle_count
    builder.add_soft_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1.0,
        vel=wp.vec3(0, 0, 0), vertices=verts, indices=indices,
        density=1e2, k_mu=mat["k_mu"], k_lambda=mat["k_lambda"], k_damp=5e-1)
    ranges[name] = (i0, builder.particle_count)
    print(f"soft: {name} ({builder.particle_count - i0} nodes)")

bone_cfg = newton.ModelBuilder.ShapeConfig(
    density=1000.0, ke=CONTACT_KE, kd=CONTACT_KD, kf=CONTACT_KF, mu=CONTACT_MU)
for name in BONES:
    try:
        bm = pv.read(f"{name}.stl").clean().triangulate()
    except FileNotFoundError:
        print(f"SKIPPED missing {name}.stl")
        continue
    faces = bm.faces.reshape(-1, 4)[:, 1:].astype(np.int32).flatten()
    mesh = newton.Mesh(np.asarray(bm.points, dtype=np.float32), faces)
    builder.add_shape_mesh(body=-1, mesh=mesh, cfg=bone_cfg)
    print(f"rigid static: {name} ({bm.n_points} verts)")

needle_cfg = newton.ModelBuilder.ShapeConfig(
    density=1000.0, ke=CONTACT_KE, kd=CONTACT_KD, kf=CONTACT_KF, mu=CONTACT_MU)
b = builder.add_body(xform=wp.transform(
    wp.vec3(float(needle_start_np[0]), float(needle_start_np[1]), float(needle_start_np[2])),
    wp.quat_identity()))
builder.add_shape_capsule(body=b, radius=3.0, half_height=12.0, cfg=needle_cfg)

builder.color()
model = builder.finalize()
model.soft_contact_ke = CONTACT_KE
model.soft_contact_kd = CONTACT_KD
model.soft_contact_kf = CONTACT_KF
model.soft_contact_mu = CONTACT_MU

solver = newton.solvers.SolverVBD(model=model, iterations=10,
    particle_enable_self_contact=True, particle_enable_tile_solve=False,
    rigid_contact_hard=False, rigid_body_particle_contact_buffer_size=8192)
s0, s1 = model.state(), model.state()
control, contacts = model.control(), model.contacts()

drive = wp.array(np.array([[0, 0, sgn * NEEDLE_SPEED, 0, 0, 0]], dtype=np.float32),
                 dtype=wp.spatial_vector)

viewer = newton.viewer.ViewerUSD(output_path=OUT, fps=60)
viewer.set_model(model)

p_rest = s0.particle_q.numpy().copy()
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
        tip_z = bq[0, 2] + sgn * 15.0
        depth = max(0.0, (top_z - tip_z) if sgn < 0 else (tip_z - bot_z))
        per = "  ".join(
            f"{n}={np.linalg.norm(p[a:z2] - p_rest[a:z2], axis=1).max():5.2f}mm"
            for n, (a, z2) in ranges.items())
        print(f"frame {frame:3d}  tip_depth={depth:6.2f}mm  contacts={nc:5d}  "
              f"{per}  finite={np.isfinite(p).all()}")
viewer.close()
print(f"Saved {OUT}")
