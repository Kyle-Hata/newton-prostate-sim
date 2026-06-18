import warp as wp, newton, numpy as np

builder = newton.ModelBuilder()
builder.gravity = 0.0
organs = {
    "prostate_NEW": dict(k_mu=1.03e4, k_lambda=9.3e4),
    "bladder_NEW":  dict(k_mu=5.9e3,  k_lambda=5.3e4),
}
ranges={}; start=0
pn = np.load("/workspace/tet_nodes_prostate_NEW.npy")
bn = np.load("/workspace/tet_nodes_bladder_NEW.npy")
shift = (bn.mean(0) - pn.mean(0)) * 0.7
print("shifting prostate by", np.round(shift,1), "to force overlap")

for name, mat in organs.items():
    nodes = np.load(f"/workspace/tet_nodes_{name}.npy")
    if name=="prostate_NEW":
        nodes = nodes + shift
    elems = np.load(f"/workspace/tet_elems_{name}.npy")
    verts=[wp.vec3(float(p[0]),float(p[1]),float(p[2])) for p in nodes]
    indices=[int(i) for i in elems.flatten()]
    builder.add_soft_mesh(pos=wp.vec3(0,0,0),rot=wp.quat_identity(),scale=1.0,vel=wp.vec3(0,0,0),
        vertices=verts,indices=indices,density=1e3,k_mu=mat["k_mu"],k_lambda=mat["k_lambda"],k_damp=5e-1)
    ranges[name]=(start,start+len(verts)); start+=len(verts)

builder.color(); model=builder.finalize()
p0,p1=ranges["prostate_NEW"]; b0,b1=ranges["bladder_NEW"]

solver=newton.solvers.SolverVBD(model=model,iterations=10,
    particle_enable_self_contact=True,
    particle_self_contact_radius=2.0, particle_self_contact_margin=4.0,
    particle_collision_detection_interval=0,
    particle_enable_tile_solve=False)
s0,s1=model.state(),model.state(); control,contacts=model.control(),model.contacts()
b_start=model.particle_q.numpy()[b0:b1].mean(0)
dt=(1.0/60)/80
for frame in range(60):
    for _ in range(80):
        s0.clear_forces(); model.collide(s0,contacts); solver.step(s0,s1,control,contacts,dt); s0,s1=s1,s0
    if frame%15==0 or frame==59:
        p=s0.particle_q.numpy(); shift_now=np.linalg.norm(p[b0:b1].mean(0)-b_start)
        print(f"frame {frame:3d}  bladder_shift={shift_now:6.2f}mm  finite={np.isfinite(p).all()}")
print("done")
