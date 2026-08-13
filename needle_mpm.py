# needle_mpm.py: prostate as an ELASTIC MPM particle blob, needle as a
# kinematic rigid collider. This is the meshfree route, and the one that can
# eventually do real material separation (cutting).
#
# Grounded in two shipped examples at newton/examples/mpm/:
#   example_mpm_beam_twist.py       elastic material via model.mpm.young_modulus,
#                                   poisson_ratio, damping, tensile_yield_ratio
#   example_mpm_twoway_coupling.py  rigid shapes as MPM colliders via
#                                   solver.setup_collider(model=rigid_model),
#                                   with body_q/body_qd mirrored onto the MPM state
#
# Verified in a CPU sandbox: this exact construction (register attrs, particles,
# collider setup, stepping) initializes and runs stable. NOT yet verified: the
# needle actually pushing particles while moving. That is the first thing your
# run will tell us, so paste the full output including any traceback.
#
# Cutting lead, for later: tensile_yield_ratio=1.0 keeps the material fully
# elastic. Lowering it lets the material yield in tension, which is the door
# to actual tearing in MPM. Unexplored, do not present as working yet.

import numpy as np, warp as wp, newton, pyvista as pv
from newton.solvers import SolverImplicitMPM

NEEDLE_SPEED = 10.0    # mm/s downward
FRAMES = 150
FPS = 60
SUBSTEPS = 4           # MPM examples run few substeps; the solver is implicit
VOXEL = 2.5            # MPM grid voxel size in mm
SPACING = VOXEL / 2.0  # particle sampling spacing
DENSITY = 1e3

# sample particles filling the prostate volume
surf = pv.read("Prostate.stl").clean().triangulate()
lo = np.array(surf.bounds[0::2]); hi = np.array(surf.bounds[1::2])
axes = [np.arange(lo[i] + SPACING / 2, hi[i], SPACING) for i in range(3)]
gx, gy, gz = np.meshgrid(*axes, indexing="ij")
cand = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
cloud = pv.PolyData(cand)
inside = cloud.select_enclosed_points(surf, check_surface=False)["SelectedPoints"].astype(bool)
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
model = builder.finalize()
model.set_gravity((0.0, 0.0, 0.0))
model.mpm.young_modulus.fill_(3.0e4)   # prostate ~30 kPa number, same unit caveat as everywhere
model.mpm.poisson_ratio.fill_(0.45)
model.mpm.damping.fill_(1.0)
model.mpm.tensile_yield_ratio.fill_(1.0)

# needle lives in its own rigid model, registered as an MPM collider
top_z0 = float(hi[2]); cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
rb = newton.ModelBuilder()
b = rb.add_body(xform=wp.transform(wp.vec3(cx, cy, top_z0 + 17.0), wp.quat_identity()))
rb.add_shape_capsule(body=b, radius=3.0, half_height=12.0)
rigid_model = rb.finalize()
rigid_state = rigid_model.state()

opts = SolverImplicitMPM.Config()
opts.voxel_size = VOXEL
solver = SolverImplicitMPM(model, opts)
solver.setup_collider(model=rigid_model)

s0, s1 = model.state(), model.state()
for st in (s0, s1):
    st.body_q = wp.clone(rigid_state.body_q)
    st.body_qd = wp.clone(rigid_state.body_qd)

drive = wp.array(np.array([[0, 0, -NEEDLE_SPEED, 0, 0, 0]], dtype=np.float32),
                 dtype=wp.spatial_vector)

viewer = newton.viewer.ViewerUSD(output_path="/workspace/needle_mpm.usd", fps=FPS)
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
print("Saved /workspace/needle_mpm.usd")
