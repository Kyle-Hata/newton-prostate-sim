import trimesh
import sys

path = sys.argv[1] if len(sys.argv) > 1 else None

if path:
    mesh = trimesh.load(path)
    print("Loaded:", path)
else:
    mesh = trimesh.creation.torus(major_radius=1.0, minor_radius=0.3)
    print("No file given, using a generated torus")

print("Vertices:", len(mesh.vertices))
print("Faces:", len(mesh.faces))
print("Watertight?", mesh.is_watertight)
print("Volume:", mesh.volume if mesh.is_watertight else "N/A (not watertight)")
print("Euler number:", mesh.euler_number)
