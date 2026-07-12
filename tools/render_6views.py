"""Blender script: load S4 placement and render 6 camera views.
Usage: blender --background --python tools/render_6views.py -- <placement_json> <output_dir> [--resolution 512]
"""
import bpy, json, sys, os, math, csv, argparse, mathutils

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("placement_json", type=str)
    p.add_argument("output_dir", type=str)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--csv", type=str, default="asset_data/imaginarium_asset_info.csv")
    return p.parse_args(argv)


def load_fbx_map(csv_path):
    """Map asset_id -> fbx_path."""
    fbx_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fbx_map[row["id"]] = row.get("fbx", row.get("fbx_path", ""))
    return fbx_map


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def load_scene(placement_json, fbx_map):
    """Load FBX assets into scene at placement positions."""
    data = json.load(open(placement_json, "r"))
    obj_info = data.get("obj_info", data.get("objects", {}))
    
    # Sort by dependency: floor/walls first, then objects without parent, then children
    deferred = []  # objects whose parent doesn't exist yet
    
    for obj_name, info in obj_info.items():
        asset_id = info.get("retrieved_asset", "")
        pose = info.get("pose_matrix_for_blender")
        
        if obj_name.startswith(("floor_","wall_","ceiling_")):
            # Create simple plane/cube for structural elements
            if obj_name.startswith("floor_"):
                bpy.ops.mesh.primitive_plane_add(size=10)
                obj = bpy.context.object
                obj.name = obj_name
                if pose:
                    obj.matrix_world = mathutils.Matrix(pose)
            elif obj_name.startswith("wall_"):
                bpy.ops.mesh.primitive_cube_add(size=1)
                obj = bpy.context.object
                obj.name = obj_name
                if pose:
                    obj.matrix_world = mathutils.Matrix(pose)
            continue
        
        if not asset_id or asset_id not in fbx_map:
            # create placeholder cube
            bpy.ops.mesh.primitive_cube_add(size=0.3)
            obj = bpy.context.object
            obj.name = obj_name
            if pose:
                obj.matrix_world = mathutils.Matrix(pose)
            continue
        
        fbx_path = fbx_map[asset_id]
        if not os.path.exists(fbx_path):
            bpy.ops.mesh.primitive_cube_add(size=0.3)
            obj = bpy.context.object
            obj.name = obj_name
            if pose:
                obj.matrix_world = mathutils.Matrix(pose)
            continue
        
        try:
            bpy.ops.import_scene.fbx(filepath=fbx_path)
            obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
            if obj:
                obj.name = obj_name
                if pose:
                    obj.matrix_world = mathutils.Matrix(pose)
        except Exception as e:
            print(f"  WARN: failed to load {fbx_path}: {e}")
    
    # Set up lighting
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.object
    sun.data.energy = 3.0
    
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.film_transparent = True


def render_views(output_dir, scene_name):
    """Render 6 fixed camera views."""
    resolution = bpy.context.scene.render.resolution_x
    views = [
        ("front",  (8, 0, 2),   mathutils.Euler((math.radians(90), 0, math.radians(-90)))),
        ("back",   (-8, 0, 2),  mathutils.Euler((math.radians(90), 0, math.radians(90)))),
        ("left",   (0, -8, 2),  mathutils.Euler((math.radians(90), 0, math.radians(180)))),
        ("right",  (0, 8, 2),   mathutils.Euler((math.radians(90), 0, 0))),
        ("top",    (0, 0, 10),  mathutils.Euler((0, 0, 0))),
        ("iso",    (6, -6, 5),  mathutils.Euler((math.radians(60), 0, math.radians(-45)))),
    ]
    
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    
    for view_name, loc, rot in views:
        bpy.ops.object.camera_add(location=loc, rotation=rot)
        cam = bpy.context.object
        cam.name = f"cam_{view_name}"
        cam.data.lens = 35
        bpy.context.scene.camera = cam
        
        out_path = os.path.join(output_dir, f"{scene_name}_{view_name}.png")
        bpy.context.scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        print(f"  rendered: {out_path}")
        
        # remove camera
        bpy.data.objects.remove(cam, do_unlink=True)


def main():
    args = parse_args()
    scene_name = os.path.basename(args.placement_json).split("_placement_info")[0]
    
    print(f"Loading {args.placement_json}")
    clear_scene()
    
    fbx_base = "asset_data/imaginarium_assets"
    load_scene(args.placement_json, fbx_map)
    
    os.makedirs(args.output_dir, exist_ok=True)
    render_views(args.output_dir, scene_name)
    print("Done.")


if __name__ == "__main__":
    main()
