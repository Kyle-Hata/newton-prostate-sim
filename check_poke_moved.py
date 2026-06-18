import warp as wp, newton, numpy as np
builder = newton.ModelBuilder(); builder.gravity = 0.0
organs = {"prostate_NEW": dict(k_mu=1.03e4,k_lambda=9.3e4),
          "bladder_NEW": dict(k_mu=5.9e3,k_lambda=5.3e4)}
ranges={}; start=0
for name,mat in organs.items():
    nodes=np.load(f"/workspace/tet_nodes_{name}.npy"); elems=np.load(f"/workspace/tet_elems_{name}.npy")
    verts=[wp.vec3(float(p[0]),float(p[1]),float(p[2])) for p in nodes]
    indices=[int(i) for i in elems.flatten()]
    builder.add_soft_mesh(pos=wp.vec3(0,0,0),rot=wp.quat_identity(),scale=1.0,vel=wp.vec3(0,0,0),
        vertices=verts,indices=indices,density=1e3,k_mu=mat["k_mu"],k_lambda=mat["k_lambda"],k_damp=5e-1)
    ranges[name]=(start,start+len(verts)); start+=len(verts)
builder.color(); model=builder.finalize()
qs=model.particle_q.numpy(); vel=model.particle_qd.numpy()
p0,p1=ranges["prostate_NEW"]; pros=qs[p0:p1]
top_y=pros[:,1].max(); top=np.where(pros[:,1]>top_y-5.0)[0]+p0
vel[top,1]=-5.0; model.particle_qd=wp.array(vel,dtype=wp.vec3)
before=qs[top].copy()
solver=newton.solvers.SolverVBD(model=model,iterations=10,particle_enable_self_contact=False,particle_enable_tile_solve=False)
s0,s1=model.state(),model.state(); control,contacts=model.control(),model.contacts()
dt=(1.0/60)/80
for f in range(120):
    for _ in range(80):
        s0.clear_forces(); model.collide(s0,contacts); solver.step(s0,s1,control,contacts,dt); s0,s1=s1,s0
after=s0.particle_q.numpy()[top]
moved=np.linalg.norm(after-before,axis=1)
print(f"poked nodes moved: mean={moved.mean():.2f}mm  max={moved.max():.2f}mm  min={moved.min():.2f}mm")
