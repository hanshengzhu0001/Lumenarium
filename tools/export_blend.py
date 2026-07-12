"""Generate .blend files from S4 placement data. Usage from repo root:
blender --background --python tools/export_blend.py -- <placement_json> <output_dir> [--scene-name NAME]
"""
import bpy, json, os, sys, math, argparse, mathutils

def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("placement_json")
    p.add_argument("output_dir")
    p.add_argument("--scene-name", default="")
    return p.parse_args(argv)

def main():
    args = parse_args()
    scene_name = args.scene_name or os.path.basename(args.placement_json).split("_placement_info")[0]
    fbx_base = "asset_data/imaginarium_assets"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clear
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Load placement
    data = json.load(open(args.placement_json))
    objs = data.get("obj_info", {})
    print(f"Loading {len(objs)} objects into {scene_name}")
    
    # Collections
    for col_name in ["floor_wall", "furniture"]:
        if col_name not in bpy.data.collections:
            bpy.data.collections.new(col_name)
            bpy.context.scene.collection.children.link(bpy.data.collections[col_name])
    
    for obj_name, info in objs.items():
        is_structural = obj_name.startswith(("floor_", "wall_", "ceiling_"))
        target_col = "floor_wall" if is_structural else "furniture"
        
        pose = info.get("pose_matrix_for_blender")
        asset_id = info.get("retrieved_asset", "")
        
        if is_structural:
            if "floor" in obj_name:
                # Floor: 10x10x0.04 cuboid (matches S4 pipeline)
                bpy.ops.mesh.primitive_cube_add(size=1)
                obj = bpy.context.object
                obj.name = obj_name
                obj.scale = (10, 10, 0.04)
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            elif obj_name.startswith("ceiling_"):
                # Ceiling: 10x10x0.04 cuboid
                bpy.ops.mesh.primitive_cube_add(size=1)
                obj = bpy.context.object
                obj.name = obj_name
                obj.scale = (10, 10, 0.04)
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            else:
                # Wall: 10x10x0.04 cuboid (matches S4 pipeline)
                bpy.ops.mesh.primitive_cube_add(size=1)
                obj = bpy.context.object
                obj.name = obj_name
                obj.scale = (10, 10, 0.04)
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            # Material (cream/wood) for visibility
            def get_mat(n, col):
                if n in bpy.data.materials: return bpy.data.materials[n]
                m = bpy.data.materials.new(n)
                m.use_nodes = True
                bsdf = m.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
                bsdf.inputs['Base Color'].default_value = col
                return m
            if "floor" in obj_name:
                obj.data.materials.clear()
                obj.data.materials.append(get_mat("WALL_FLOOR", (0.92, 0.88, 0.82, 1)))
            elif obj_name.startswith("ceiling_"):
                obj.data.materials.clear()
                obj.data.materials.append(get_mat("WALL_CEIL", (0.95, 0.95, 0.92, 1)))
            else:
                obj.data.materials.clear()
                obj.data.materials.append(get_mat("WALL_TEX", (0.85, 0.78, 0.70, 1)))
            if pose: obj.matrix_world = mathutils.Matrix(pose)
        else:
            fbx_path = os.path.join(fbx_base, f"{asset_id}.fbx") if asset_id else None
            if fbx_path and os.path.exists(fbx_path):
                try:
                    bpy.ops.import_scene.fbx(filepath=fbx_path)
                    obj = bpy.context.selected_objects[0]
                    obj.name = obj_name
                    if pose: obj.matrix_world = mathutils.Matrix(pose)
                except:
                    bpy.ops.mesh.primitive_cube_add(size=0.3)
                    obj = bpy.context.object
                    obj.name = obj_name
                    if pose: obj.matrix_world = mathutils.Matrix(pose)
            else:
                bpy.ops.mesh.primitive_cube_add(size=0.3)
                obj = bpy.context.object
                obj.name = obj_name
                if pose: obj.matrix_world = mathutils.Matrix(pose)
        
        # Move to collection
        for col in obj.users_collection:
            col.objects.unlink(obj)
        bpy.data.collections[target_col].objects.link(obj)
    
    # Lighting
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 8))
    bpy.context.object.data.energy = 2.5
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 6))
    bpy.context.object.data.energy = 50
    
    # Camera - front-ish view as default
    bpy.ops.object.camera_add(location=(4, -6, 2), rotation=(math.radians(80), 0, math.radians(45)))
    bpy.context.scene.camera = bpy.context.object
    
    # Render settings
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    
    # Render preview
    bpy.context.scene.render.filepath = os.path.join(args.output_dir, f"{scene_name}_preview.png")
    print(f"  preview: {scene_name}_preview.png")
    
    # Save .blend
    blend_path = os.path.join(args.output_dir, f"{scene_name}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"  blend: {blend_path}")
    
    # Auto-render 3 key views
    views = [
        ("front", (6, 0, 1.5), mathutils.Euler((math.radians(90), 0, math.radians(-90)))),
        ("persp", (5, -5, 3), mathutils.Euler((math.radians(65), 0, math.radians(-45)))),
        ("top",   (0, 0, 8), mathutils.Euler((0, 0, 0))),
    ]
    for vn, loc, rot in views:
        bpy.ops.object.camera_add(location=loc, rotation=rot)
        cam = bpy.context.object; bpy.context.scene.camera = cam
        bpy.context.scene.render.filepath = os.path.join(args.output_dir, f"{scene_name}_{vn}.png")
        bpy.data.objects.remove(cam, do_unlink=True)
    
    print(f"  Done: {scene_name}")

if __name__ == "__main__":
    main()
