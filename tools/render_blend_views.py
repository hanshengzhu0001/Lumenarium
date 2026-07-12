import bpy, os, sys, math, mathutils, random

blend_path = sys.argv[-1]
out_dir = os.path.dirname(blend_path)
scene_name = os.path.basename(blend_path).replace(".blend", "")

bpy.ops.wm.open_mainfile(filepath=blend_path)
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'

# Compute scene bounds
all_bounds = []
for o in bpy.data.objects:
    if o.type == 'MESH' and o.data.vertices:
        bbox = [o.matrix_world @ mathutils.Vector(c) for c in o.bound_box]
        for v in bbox: all_bounds.append(v)
if not all_bounds: 
    print("No meshes"); sys.exit(1)

mins = [min(v[i] for v in all_bounds) for i in range(3)]
maxs = [max(v[i] for v in all_bounds) for i in range(3)]
center = [(mins[i]+maxs[i])/2 for i in range(3)]
size = max(maxs[i]-mins[i] for i in range(3))
print(f"Scene bounds: size={size:.1f}m, center={center}")

# Scale down the floor (it's a giant 10m plane)
for o in bpy.data.objects:
    if o.type == 'MESH' and 'floor' in o.name.lower():
        # Make it just slightly bigger than objects
        scale = size * 1.3
        o.scale = (scale, scale, 1)
    if o.type == 'MESH' and 'wall' in o.name.lower() and not 'wall_mounted' in o.name:
        # Make walls reasonable
        o.scale = (size/5, size/5, size/3)

# Force material on every mesh
palette = [
    (0.85, 0.45, 0.35, 1), (0.45, 0.65, 0.85, 1), (0.55, 0.75, 0.55, 1),
    (0.95, 0.85, 0.45, 1), (0.65, 0.45, 0.85, 1), (0.85, 0.65, 0.85, 1),
    (0.45, 0.85, 0.85, 1), (0.85, 0.55, 0.65, 1), (0.65, 0.65, 0.65, 1),
    (0.5, 0.5, 0.5, 1)
]
random.seed(42)
for i, o in enumerate(bpy.data.objects):
    if o.type != 'MESH': continue
    if not o.data.materials or not o.data.materials[0]:
        mat = bpy.data.materials.new(name=f"mat_{i}")
        mat.use_nodes = False
        if 'floor' in o.name.lower(): mat.diffuse_color = (0.7, 0.65, 0.55, 1)
        elif 'wall' in o.name.lower(): mat.diffuse_color = (0.85, 0.83, 0.78, 1)
        else: mat.diffuse_color = palette[i % len(palette)]
        o.data.materials.append(mat)

# Lights - place at scene center + offset
bpy.ops.object.select_all(action='DESELECT')
for o in list(bpy.data.objects):
    if o.type == 'LIGHT': 
        o.select_set(True)
        bpy.ops.object.delete()
bpy.context.scene.world.use_nodes = False
bpy.context.scene.world.color = (0.5, 0.5, 0.5)

bpy.ops.object.light_add(type='SUN', location=(center[0], center[1]-size, center[2]+size))
bpy.context.object.data.energy = 5
bpy.ops.object.light_add(type='AREA', location=(center[0]+size, center[1], center[2]+size))
bpy.context.object.data.energy = 500
bpy.context.object.data.size = size
bpy.ops.object.light_add(type='AREA', location=(center[0]-size, center[1], center[2]+size))
bpy.context.object.data.energy = 500
bpy.context.object.data.size = size

try: bpy.context.scene.eevee.taa_render_samples = 1
except: pass
bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 768

views = [
    ("front",  (center[0]+size*1.5, center[1], center[2]), (math.radians(82), 0, math.radians(-90))),
    ("back",   (center[0]-size*1.5, center[1], center[2]), (math.radians(82), 0, math.radians(90))),
    ("left",   (center[0], center[1]-size*1.5, center[2]), (math.radians(82), 0, math.radians(180))),
    ("right",  (center[0], center[1]+size*1.5, center[2]), (math.radians(82), 0, 0)),
    ("top",    (center[0], center[1], center[2]+size*2.5), (0, 0, 0)),
    ("persp",  (center[0]+size, center[1]-size, center[2]+size*0.8), (math.radians(60), 0, math.radians(-45))),
]

for vn, loc, rot in views:
    bpy.ops.object.camera_add(location=loc, rotation=rot)
    cam = bpy.context.object
    cam.data.lens = 35
    bpy.context.scene.camera = cam
    bpy.context.scene.render.filepath = os.path.join(out_dir, f"{scene_name}_{vn}.png")
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)
    print(f"  {vn}", flush=True)
print("Done")
