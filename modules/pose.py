"""
Module: Transformations Estimation (Step 8 - Pose Estimation)
模块：变换估计 (步骤 8 - 姿态估计)

Fully migrated from S3_pose_inference_op.py
"""
import os
import json
import re

from core.context import Context
from utils.io import load_json, save_json

STACKABLE_UPPER_HINTS = {
    "box", "crate", "carton", "container", "case", "toolbox"
}
STACK_SUPPORT_HINTS = {
    "pallet", "crate", "box", "carton", "shelf", "rack", "cabinet", "table", "desk", "bench", "workbench"
}
STACK_NEVER_UPPER_HINTS = {
    "carpet", "rug", "curtain", "window", "map", "sign", "billboard", "picture", "frame",
    "mirror", "lamp", "chandelier", "chair", "sofa", "stool", "cabinet", "desk", "table",
    "shelf", "rack", "workbench", "washing", "machine", "refrigerator", "stove", "oven",
    "plant", "potted", "faucet", "hose",
}


def _name_tokens(name):
    return {part for part in re.split(r"[^a-z0-9]+", str(name or "").lower()) if part}


def _has_any(tokens, hints):
    return bool(tokens & hints)


def _is_anonymous_object(name):
    return re.match(r"^object(?:_\d+)+$", str(name or "")) is not None


def _placement_size(name, placement_info):
    obj_info = placement_info.get("obj_info", {}) if isinstance(placement_info, dict) else {}
    info = obj_info.get(name, {}) if isinstance(obj_info, dict) else {}
    size = info.get("pcd_obb_size") or info.get("bbox_size") or info.get("scale")
    if not isinstance(size, (list, tuple)) or len(size) < 3:
        return None
    try:
        vals = [abs(float(v)) for v in size[:3]]
    except Exception:
        return None
    return vals


def _object_tokens(name, scene_graph_result, placement_info=None):
    tokens = set(_name_tokens(name))
    graph_props = scene_graph_result.get(name, {}) if isinstance(scene_graph_result, dict) else {}
    if isinstance(graph_props, dict):
        for key in ("class", "category", "caption", "description", "name"):
            tokens.update(_name_tokens(graph_props.get(key)))

    obj_info = {}
    if isinstance(placement_info, dict):
        obj_info = placement_info.get("obj_info", {}).get(name, {})
    if isinstance(obj_info, dict):
        for key in ("retrieved_asset", "retrieved_asset_name", "fbx_name", "category"):
            tokens.update(_name_tokens(obj_info.get(key)))
    return tokens


def _should_apply_stacking_pair(lower, upper, scene_graph_result, placement_info=None):
    """Conservative gate for v3 geometric stacking re-parenting.

    The detector is intentionally broad, so the re-parent step must be narrow:
    only small cargo-like objects may be lifted from the floor onto clear
    support-like objects. Existing non-floor parents are kept unless they
    already agree with the proposed lower object.
    """
    upper_props = scene_graph_result.get(upper, {}) if isinstance(scene_graph_result, dict) else {}
    lower_props = scene_graph_result.get(lower, {}) if isinstance(scene_graph_result, dict) else {}
    if not isinstance(upper_props, dict) or not isinstance(lower_props, dict):
        return False, "missing_graph_props"
    if _is_anonymous_object(upper) or _is_anonymous_object(lower):
        return False, "anonymous_pair"

    obj_info = placement_info.get("obj_info", {}) if isinstance(placement_info, dict) else {}
    if placement_info is not None and (upper not in obj_info or lower not in obj_info):
        return False, "missing_placement_obj"

    upper_tokens = _object_tokens(upper, scene_graph_result, placement_info)
    lower_tokens = _object_tokens(lower, scene_graph_result, placement_info)
    upper_size = _placement_size(upper, placement_info)

    if _has_any(upper_tokens, STACK_NEVER_UPPER_HINTS):
        return False, "upper_anchor_or_large_object"
    if upper_props.get("isHangingFromCeiling") or upper_props.get("isHangingOnWall"):
        return False, "upper_hanging_or_wall"
    if str(upper_props.get("supported") or "").startswith(("wall_", "ceiling_")):
        return False, "upper_structural_parent"

    old_parent = upper_props.get("supported")
    old_is_floor = old_parent == "floor_0" or upper_props.get("isOnFloor") is True
    already_supported = old_parent == lower

    if not old_is_floor and not already_supported:
        return False, "preserve_existing_non_floor_parent"

    # Only cargo-like uppers should enter stack-aware S4. Ordinary table-top
    # objects already have a parent and do not need the strong stack lock.
    if not _has_any(upper_tokens, STACKABLE_UPPER_HINTS):
        return False, "upper_not_stackable"
    if not _has_any(lower_tokens, STACK_SUPPORT_HINTS):
        return False, "lower_not_support_like"
    if upper_size is not None:
        max_dim = max(upper_size)
        volume = upper_size[0] * upper_size[1] * upper_size[2]
        if max_dim > float(os.environ.get("IMAGINARIUM_S3_STACK_MAX_UPPER_DIM", "1.6")):
            return False, "upper_too_large_for_stack"
        if volume > float(os.environ.get("IMAGINARIUM_S3_STACK_MAX_UPPER_VOLUME", "1.5")):
            return False, "upper_volume_too_large_for_stack"

    return True, "already_supported" if already_supported else "ok"


class PoseModule:
    """
    Pose Estimation Module
    姿态估计模块
    
    Performs:
    - Rotation estimation (view matching)
    - Translation estimation (OBB alignment)
    - Scale estimation (volume optimization)
    """
    def __init__(self, context: Context):
        self.context = context
        self.logger = context.logger
        self.cfg = context.config.get('S3_pose_inference', {})
        self.shared_cfg = context.config.shared
        
    def run(self):
        """
        Main execution: Pose estimation
        """
        self.logger.info(">>> Stage 4: Pose Estimation")
        
        S1_folder = os.path.join(self.context.output_dir, 'S1_scene_parsing_results')
        S2_folder = os.path.join(self.context.output_dir, 'S2_3d_retrieval_results')
        save_dir = os.path.join(self.context.output_dir, 'S3_pose_inference')
        os.makedirs(save_dir, exist_ok=True)
        
        scene_name = self.context.image_name
        
        # Smart resume: Check if S3 placement info file exists
        placement_info_path = os.path.join(save_dir, f'{scene_name}_placement_info.json')
        
        if not self.context.clean_mode and os.path.exists(placement_info_path):
            self.logger.info(f"✓ S3 已完成：所有必需文件都存在，跳过此阶段")
            self.logger.info(f"  - {scene_name}_placement_info.json: ✓")
            # Store path in context for next stage
            self.context.set_data('placement_info_path', placement_info_path)
            self.logger.info("Pose Estimation Done (Skipped, placement info file exists).")
            return
        
        # 1. Load Data
        depth_image_path = os.path.join(self.context.output_dir, 'S0_geometry_pred_results/depth.png')
        retrieval_dict = self.context.get_data('retrieval_results')
        if not retrieval_dict:
             retrieval_dict = load_json(os.path.join(S2_folder, 'retrieval_results_final.json'))
             
        obb_data_path = os.path.join(S2_folder, 'pcd_obb_data.json')
        loaded_obb_data = load_json(obb_data_path)

        from modules._s3_legacy_functions import (
            detect_truncated_objects,
            combine_scene_objects_pose,
            inference_obj_pose as s3_inference_obj_pose,
            detect_stacking_pairs,
            verify_parent_with_obb,
        )
        
        # 2. Load Shared AENet Model (reuses DINOv2 from S2)
        ae_net = self.context.get_ae_net(self.cfg.ae_net_weights_path)
        
        # 3. Set legacy globals & Patch model loader
        import modules._s3_legacy_functions as s3_legacy
        s3_legacy.logger = self.logger
        
        # Monkey-patch load_ae_net to return our shared model
        original_load_ae_net = s3_legacy.load_ae_net if hasattr(s3_legacy, 'load_ae_net') else None
        s3_legacy.load_ae_net = lambda *args, **kwargs: ae_net
        
        try:
            # 4. Run Inference
            self.logger.info("Running pose inference with shared AE Net...")
            predictions_id_result, comparison_images = s3_inference_obj_pose(
                S1_folder, 
                self.cfg.template_dir, 
                depth_image_path, 
                retrieval_dict, 
                loaded_obb_data, 
                self.cfg.ae_net_weights_path,  # Path (ignored by patch)
                self.cfg.ori_dino_weights_path,  # Path (ignored by patch)
                save_dir,
                use_homography=self.cfg.get('use_homography', True),
                save_pts_match_imgs=self.context.debug_mode,
                save_comparison_imgs=self.context.debug_mode
            )
        finally:
            # Restore original
            if original_load_ae_net:
                s3_legacy.load_ae_net = original_load_ae_net
        
        # 5. Save & Vis
        if comparison_images:
            from utils.image_concat import stitch_images_grid
            stitch_images_grid(save_dir, os.path.join(save_dir, 'pose_prediction_stitched.png'), comparison_images)
            
        # 6. Placement Info
        wall_floor_pose = load_json(os.path.join(S1_folder, 'floor_walls_pose.json'))
        scene_graph_result = load_json(os.path.join(S1_folder, 'scene_graph_result_final.json'))
        truncated_info = detect_truncated_objects(S1_folder)
        
        save_path = os.path.join(save_dir, f'{scene_name}_placement_info.json')
        combine_scene_objects_pose(
            predictions_id_result, 
            wall_floor_pose, 
            scene_graph_result, 
            retrieval_dict, 
            truncated_info, 
            save_path
        )
        
        # v3: 堆叠感知 — 检测落地物体之间的堆叠关系并写入 placement_info
        if os.environ.get('IMAGINARIUM_S3_STACK_AWARE', '0') == '1':
            # v4: OBB proximity 验证 VLM parent (论文 Algorithm 2)
            scene_graph_result = verify_parent_with_obb(S1_folder, scene_graph_result)
            stacking_pairs = detect_stacking_pairs(S1_folder, scene_graph_result)
            if stacking_pairs:
                self.logger.info(f"[v3] 检测到 {len(stacking_pairs)} 组堆叠关系: {stacking_pairs}")
                # 写入 placement_info.json
                placement_info = load_json(save_path)
                placement_info['stacking_pairs_raw'] = [[lower, upper] for lower, upper in stacking_pairs]
                accepted_pairs = []
                rejected_pairs = []
                # 在 obj_info 里标记 upper 物体的 stacked_on + 修正 supported
                for lower, upper in stacking_pairs:
                    if upper in placement_info['obj_info']:
                        should_apply, reason = _should_apply_stacking_pair(
                            lower, upper, scene_graph_result, placement_info
                        )
                        if not should_apply:
                            rejected_pairs.append([lower, upper, reason])
                            self.logger.info(f"[v3] 跳过不可靠堆叠: {upper} on {lower} ({reason})")
                            continue
                        accepted_pairs.append([lower, upper])
                        placement_info['obj_info'][upper]['stacked_on'] = lower
                        # v3 几何检测到的堆叠关系比 GPT 的 supported 更可靠
                        # 无条件覆盖：若 GPT 已经是正确值则 no-op，否则修正
                        old_sup = placement_info['obj_info'][upper].get('supported', '')
                        if old_sup != lower:
                            placement_info['obj_info'][upper]['supported'] = lower
                            self.logger.info(f"[v3] 修正场景图: {upper}.supported: {old_sup} → {lower}")
                placement_info['stacking_pairs'] = accepted_pairs
                placement_info['stacking_pairs_rejected'] = rejected_pairs
                import json as _json
                with open(save_path, 'w') as _f:
                    _json.dump(placement_info, _f, indent=2)
                self.logger.info(f"[v3] 堆叠关系已写入 placement_info: {save_path}")
            else:
                self.logger.info("[v3] 未检测到堆叠关系")
        
        self.context.set_data('placement_info_path', save_path)
        self.logger.info(f"Pose Estimation Done. Saved to {save_path}")
        
        # 7. Cleanup S3 resources to free VRAM for Blender (S4)
        self.context.release_models()
