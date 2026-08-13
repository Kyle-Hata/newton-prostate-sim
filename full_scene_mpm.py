# full_scene_mpm.py: ALL organs + bones + needle, meshfree MPM particles.
#
# Per-organ stiffness via per-particle young_modulus fills, following the
# pattern in example_mpm_beam_twist.py (indexed fill_ on model.mpm arrays).
# Bones and needle are shapes in the same model, registered as colliders via
# solver.setup_collider(model=model), same construction your needle_mpm_v3
# verified on the GPU. Organ-organ interaction comes free from the shared MPM
# grid, with one caveat worth knowing: touching organs behave near-welded at
# their contact, since MPM has no slip interface between materials.
#
# Rendering uses the v3 recipe that worked in your Isaac: per-organ log_points
# plus post-conversion of each Points prim to a PointInstancer of spheres.
#
# Sampling is 1.5mm here (vs 1.25 for prostate-only) to keep total particle
# count sane across four organs. First run: consider FRAMES = 30.

import numpy as np, warp as wp, newton, pyvista as pv
from newton.solvers import SolverImplicitMPM

NEEDLE_SPEED = 10.0
FRAMES = 150
FPS = 60
SUBSTEPS = 4
VOXEL = 3.0
SPACING = VOXEL / 2.0
DENSITY = 1e3
OUT = "/workspace/full_scene_mpm.usd"

SOFT = {
    "Prostate": dict(E=3.0e4,  color=(0.90, 0.55, 0.45)),
    "Bladder":  dict(E=1.7e4,  color=(0.95, 0.80, 0.40)),
    "Rectum":   dict(E=2.3e3,  color=(0.70, 0.50, 0.60)),
    "Urethra":  dict(E=3.07e4, color=(0.40, 0.75, 0.65)),
}
BONES = ["Bone_1", "Bone_2"]

surfs = {name: pv.read(f"{name}.stl").clean().triangulate() for name in SOFT}

pro = surfs["Prostate"]
ctr = np.array(pro.center)
top_z, bot_z = float(pro.bounds[5]), float(pro.bounds[4])
candidates = {
    "superior descending":     (np.array([ctr[0], ctr[1], top_z + 17.0]), -1.0),
    "transperineal ascending": (np.array([ctr[0], ctr[1], bot_z - 17.0]), +1.0),
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

def sample_inside(surf):
    lo = np.array(surf.bounds[0::2]); hi = np.array(surf.bounds[1::2])
    axes = [np.arange(lo[i] + SPACING / 2, hi[i], SPACING) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    cand = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    keep = pv.PolyData(cand).select_enclosed_points(
        surf, check_surface=False)["SelectedPoints"].astype(bool)
    return cand[keep]

mass = float(SPACING ** 3 * DENSITY)
builder = newton.ModelBuilder()
SolverImplicitMPM.register_custom_attributes(builder)

ranges = {}
for name in SOFT:
    pts = sample_inside(surfs[name])
    i0 = builder.particle_count
    builder.add_particles(
        pos=[wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in pts],
        vel=[wp.vec3(0.0, 0.0, 0.0)] * len(pts),
        mass=[mass] * len(pts),
        radius=[SPACING * 0.5] * len(pts))
    ranges[name] = (i0, builder.particle_count)
    print(f"soft: {name} ({len(pts)} particles)")

shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
for name in BONES:
    try:
        bm = pv.read(f"{name}.stl").clean().triangulate()
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
model.set_gravity((0.0, 0.0, 0.0))
model.mpm.poisson_ratio.fill_(0.45)
model.mpm.damping.fill_(1.0)
model.mpm.tensile_yield_ratio.fill_(1.0)
for name, (a, z2) in ranges.items():
    model.mpm.young_modulus[a:z2].fill_(SOFT[name]["E"])

opts = SolverImplicitMPM.Config()
opts.voxel_size = VOXEL
solver = SolverImplicitMPM(model, opts)
solver.setup_collider(model=model)

s0, s1 = model.state(), model.state()
drive = wp.array(np.array([[0, 0, sgn * NEEDLE_SPEED, 0, 0, 0]], dtype=np.float32),
                 dtype=wp.spatial_vector)

viewer = newton.viewer.ViewerUSD(output_path=OUT, fps=FPS)
viewer.set_model(model)

p_rest = s0.particle_q.numpy().copy()
dt = 1.0 / (FPS * SUBSTEPS)
needle_z = float(needle_start_np[2])
for frame in range(FRAMES):
    for _ in range(SUBSTEPS):
        needle_z += sgn * NEEDLE_SPEED * dt
        pose = np.array([[float(needle_start_np[0]), float(needle_start_np[1]),
                          needle_z, 0, 0, 0, 1]], dtype=np.float32)
        pose_wp = wp.array(pose, dtype=wp.transform)
        for st in (s0, s1):
            wp.copy(st.body_q, pose_wp)
            wp.copy(st.body_qd, drive)
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
    viewer.begin_frame(frame / FPS)
    viewer.log_state(s0)
    for name, (a, z2) in ranges.items():
        viewer.log_points(f"{name.lower()}_particles", s0.particle_q[a:z2],
                          radii=SPACING * 0.6, colors=SOFT[name]["color"])
    viewer.end_frame()
    if frame % 15 == 0 or frame == FRAMES - 1:
        p = s0.particle_q.numpy()
        tip_z = needle_z + sgn * 15.0
        depth = max(0.0, (top_z - tip_z) if sgn < 0 else (tip_z - bot_z))
        per = "  ".join(
            f"{n}={np.linalg.norm(p[a:z2] - p_rest[a:z2], axis=1).max():5.2f}mm"
            for n, (a, z2) in ranges.items())
        print(f"frame {frame:3d}  tip_depth={depth:6.2f}mm  {per}  "
              f"finite={np.isfinite(p).all()}")
viewer.close()

# convert every organ Points prim to a PointInstancer of spheres for Isaac
from pxr import Usd, UsdGeom, Gf
stage = Usd.Stage.Open(OUT)
wanted = {f"{name.lower()}_particles": SOFT[name]["color"] for name in SOFT}
converted = 0
for prim in list(stage.Traverse()):
    tname = prim.GetTypeName()
    if tname != "Points":
        continue
    pname = prim.GetPath().name
    if pname not in wanted:
        UsdGeom.Imageable(prim).MakeInvisible()
        continue
    attr = UsdGeom.Points(prim).GetPointsAttr()
    ts = attr.GetTimeSamples()
    sample = attr.Get(ts[0]) if ts else attr.Get()
    if sample is None:
        print(f"WARNING: {pname} has no data, skipped")
        continue
    n = len(sample)
    base = f"/root/spheres_{pname}"
    inst = UsdGeom.PointInstancer.Define(stage, base)
    proto = UsdGeom.Sphere.Define(stage, base + "/proto")
    proto.GetRadiusAttr().Set(SPACING * 0.7)
    proto.GetDisplayColorAttr().Set([Gf.Vec3f(*wanted[pname])])
    inst.GetPrototypesRel().SetTargets([proto.GetPath()])
    inst.GetProtoIndicesAttr().Set([0] * n)
    for t in (ts if ts else [None]):
        p = attr.Get(t) if t is not None else attr.Get()
        if p is not None:
            if t is not None:
                inst.GetPositionsAttr().Set(p, t)
            else:
                inst.GetPositionsAttr().Set(p)
    attr.Clear()
    UsdGeom.Imageable(prim).MakeInvisible()
    converted += 1
    print(f"converted {pname}: {n} spheres")
stage.Save()
print(f"Saved {OUT} ({converted} organ sphere clouds)")
