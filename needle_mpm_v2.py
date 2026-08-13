# needle_mpm_v2.py: fixes the empty-Isaac problem from v1. Two causes, both mine:
#
# 1. v1 put the needle in a separate rigid model the viewer never saw. Now the
#    capsule lives in the SAME model as the particles, so the viewer renders it
#    and the state carries body_q natively (verified locally: capsule shows up
#    as a Mesh prim in the USD).
# 2. The viewer writes particles as a UsdGeom.Points prim, which Isaac's RTX
#    renderer often skips silently. After the sim, this script converts that
#    prim into a PointInstancer of spheres, which Isaac reliably renders
#    (conversion verified locally on a test USD, all time samples preserved).
#
# Watch item: setup_collider(model=model) now points at the same model that
# holds the particles. Smoke tests passed with a separate model; same-model is
# the one construction difference. If the run errors there, paste the traceback.

import numpy as np, warp as wp, newton, pyvista as pv
from newton.solvers import SolverImplicitMPM

NEEDLE_SPEED = 10.0
FRAMES = 150
FPS = 60
SUBSTEPS = 4
VOXEL = 2.5
SPACING = VOXEL / 2.0
DENSITY = 1e3
OUT = "/workspace/needle_mpm_v2.usd"

surf = pv.read("Prostate.stl").clean().triangulate()
lo = np.array(surf.bounds[0::2]); hi = np.array(surf.bounds[1::2])
axes = [np.arange(lo[i] + SPACING / 2, hi[i], SPACING) for i in range(3)]
gx, gy, gz = np.meshgrid(*axes, indexing="ij")
cand = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
inside = pv.PolyData(cand).select_enclosed_points(surf, check_surface=False)["SelectedPoints"].astype(bool)
pts = cand[inside]
print(f"prostate MPM particles: {len(pts)} (spacing {SPACING}mm)")

mass = float(SPACING ** 3 * DENSITY)
builder = newton.ModelBuilder()
SolverImplicitMPM.register_custom_attributes(builder)
builder.add_particles(
    pos=[wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in pts],
    vel=[wp.vec3(0.0, 0.0, 0.0)] * len(pts),
    mass=[mass] * len(pts),
    radius=[SPACING * 0.5] * len(pts))

top_z0 = float(hi[2]); cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
b = builder.add_body(xform=wp.transform(wp.vec3(cx, cy, top_z0 + 17.0), wp.quat_identity()))
builder.add_shape_capsule(body=b, radius=3.0, half_height=12.0)

model = builder.finalize()
model.set_gravity((0.0, 0.0, 0.0))
model.mpm.young_modulus.fill_(3.0e4)
model.mpm.poisson_ratio.fill_(0.45)
model.mpm.damping.fill_(1.0)
model.mpm.tensile_yield_ratio.fill_(1.0)

opts = SolverImplicitMPM.Config()
opts.voxel_size = VOXEL
solver = SolverImplicitMPM(model, opts)
solver.setup_collider(model=model)

s0, s1 = model.state(), model.state()
drive = wp.array(np.array([[0, 0, -NEEDLE_SPEED, 0, 0, 0]], dtype=np.float32),
                 dtype=wp.spatial_vector)

viewer = newton.viewer.ViewerUSD(output_path=OUT, fps=FPS)
viewer.set_model(model)

p_rest = s0.particle_q.numpy().copy()
dt = 1.0 / (FPS * SUBSTEPS)
needle_z = top_z0 + 17.0
for frame in range(FRAMES):
    for _ in range(SUBSTEPS):
        needle_z -= NEEDLE_SPEED * dt
        pose = np.array([[cx, cy, needle_z, 0, 0, 0, 1]], dtype=np.float32)
        pose_wp = wp.array(pose, dtype=wp.transform)
        for st in (s0, s1):
            wp.copy(st.body_q, pose_wp)
            wp.copy(st.body_qd, drive)
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
    viewer.begin_frame(frame / FPS); viewer.log_state(s0); viewer.end_frame()
    if frame % 15 == 0 or frame == FRAMES - 1:
        p = s0.particle_q.numpy()
        tip_z = needle_z - 15.0
        depth = max(0.0, top_z0 - tip_z)
        disp = float(np.linalg.norm(p - p_rest, axis=1).max())
        print(f"frame {frame:3d}  tip_depth={depth:6.2f}mm  max_particle_disp={disp:6.2f}mm  "
              f"finite={np.isfinite(p).all()}")
viewer.close()

# convert the Points prim to a PointInstancer of spheres so Isaac renders it
from pxr import Usd, UsdGeom, Gf
stage = Usd.Stage.Open(OUT)
pts_prim = UsdGeom.Points(stage.GetPrimAtPath("/root/model/particles"))
times = pts_prim.GetPointsAttr().GetTimeSamples() or [0.0]
inst = UsdGeom.PointInstancer.Define(stage, "/root/particle_spheres")
proto = UsdGeom.Sphere.Define(stage, "/root/particle_spheres/proto")
proto.GetRadiusAttr().Set(SPACING * 0.7)
proto.GetDisplayColorAttr().Set([Gf.Vec3f(0.9, 0.55, 0.45)])
inst.GetPrototypesRel().SetTargets([proto.GetPath()])
inst.GetProtoIndicesAttr().Set([0] * len(pts))
for t in times:
    inst.GetPositionsAttr().Set(pts_prim.GetPointsAttr().Get(t), t)
UsdGeom.Imageable(pts_prim.GetPrim()).MakeInvisible()
stage.Save()
print(f"Saved {OUT} ({len(times)} time samples, {len(pts)} sphere instances)")
