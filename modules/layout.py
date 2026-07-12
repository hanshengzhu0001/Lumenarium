import os
import subprocess
from core.context import Context

class LayoutModule:
    """
    Module: Scene Layout Optimization (Steps 9-11)
    模块：场景布局优化 (步骤 9-11)
    
    Wraps Blender Script execution.
    """
    def __init__(self, context: Context):
        self.context = context
        self.logger = context.logger
        self.cfg = context.config.get('S4_blender_layout_and_corr')

    def _run_blender(self, blender_cmd, s4_json_path):
        import time
        import threading

        self.logger.info(f"Executing Blender: {' '.join(blender_cmd)}")

        # Blender bundles torch 2.0.0, whose CUDA allocator does not fully
        # support the `expandable_segments` option. If the parent pipeline sets
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (used to mitigate OOM
        # in S1/S2/S3), inheriting it here makes torch crash with SIGSEGV during
        # CUDA init inside Blender. Strip it for the Blender subprocess only.
        blender_env = os.environ.copy()
        alloc_conf = blender_env.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" in alloc_conf:
            filtered = ",".join(
                seg for seg in alloc_conf.split(",")
                if "expandable_segments" not in seg
            )
            if filtered:
                blender_env["PYTORCH_CUDA_ALLOC_CONF"] = filtered
            else:
                blender_env.pop("PYTORCH_CUDA_ALLOC_CONF", None)

        process = subprocess.Popen(
            blender_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,  # line buffered
            cwd=os.getcwd(),
            env=blender_env
        )

        stop_indicator = threading.Event()

        def progress_indicator():
            count = 0
            while not stop_indicator.is_set():
                count += 1
                if count % 10 == 0:
                    self.logger.info(f"⏳ Blender仍在运行... ({count}秒)")
                time.sleep(1)

        progress_thread = threading.Thread(target=progress_indicator, daemon=True)
        progress_thread.start()

        try:
            for line in process.stdout:
                line = line.strip()
                if line:
                    if not any(x in line.lower() for x in ['found bundled python', 'read prefs']):
                        self.logger.info(f"[Blender] {line}")

            process.wait()
        finally:
            stop_indicator.set()
            progress_thread.join(timeout=1)

        if process.returncode != 0:
            self.logger.error("Blender process failed.")
            raise subprocess.CalledProcessError(process.returncode, blender_cmd)
        if not os.path.exists(s4_json_path):
            raise RuntimeError(f"Blender finished without S4 placement output: {s4_json_path}")
        
    def run(self):
        self.logger.info(">>> Stage 5: Layout Optimization (Blender)")
        
        # Get placement info path from previous step or construct it
        placement_json_path = self.context.get_data('placement_info_path')
        
        if not placement_json_path:
            # Fallback logic
            scene_name = self.context.image_name
            S3_folder = os.path.join(self.context.output_dir, 'S3_pose_inference')
            placement_json_path = os.path.join(S3_folder, f'{scene_name}_placement_info.json')
            
        if not os.path.exists(placement_json_path):
            raise FileNotFoundError(f"Placement info not found: {placement_json_path}")
        
        # Create S4 output folder
        S4_folder = os.path.join(self.context.output_dir, 'S4_layout_refinement')
        os.makedirs(S4_folder, exist_ok=True)
        
        # Smart resume: Check if S4 output files already exist in the new folder
        scene_name = self.context.image_name
        s4_json_path = os.path.join(S4_folder, f'{scene_name}_placement_info_s4.json')
        s4_render_path = os.path.join(S4_folder, f'{scene_name}_render_simu.png')
        
        if not self.context.clean_mode:
            if os.path.exists(s4_json_path) and os.path.exists(s4_render_path):
                self.logger.info(f"✓ S4 已完成：所有必需文件都存在，跳过此阶段")
                self.logger.info(f"  - {os.path.basename(s4_json_path)}: ✓")
                self.logger.info(f"  - {os.path.basename(s4_render_path)}: ✓")
                self.logger.info("Layout Optimization Done (Skipped, final results exist).")
                return
            
        # Path to the new blender script
        script_path = "modules/S4_blender_layout_and_corr.py"
        
        # Ensure models are released before running Blender to free VRAM
        self.context.release_models()
        
        # Blender Command
        blender_bin = os.environ.get("IMAGINARIUM_BLENDER_BIN", "blender")
        blender_cmd = [
            blender_bin,
            "--background",
            "--python", script_path,
            "--",
            "--obj_placement_info_json_path", placement_json_path,
            "--output_folder", S4_folder
        ]
        
        # 添加debug参数（如果启用）
        if self.context.debug_mode:
            blender_cmd.append("--debug")
        
        self.logger.info("⏳ 正在执行Blender摆放、逻辑优化和掉落仿真，这可能需要几分钟时间，请耐心等待...")
        
        try:
            self._run_blender(blender_cmd, s4_json_path)
                
        except Exception as e:
            self.logger.error(f"Layout Optimization Failed: {e}")
            self.logger.warning("Retrying S4 (Blender may have crashed intermittently)...")
            try:
                self._run_blender(blender_cmd, s4_json_path)
                self.logger.warning("Layout Optimization completed on retry.")
            except Exception as fallback_error:
                self.logger.error(f"Layout fallback failed: {fallback_error}")
                raise fallback_error
            
        self.logger.info("Layout Optimization Done.")
        
        # Blender process naturally releases its own resources on exit, 
        # but we can ensure the pipeline context is clean if needed.
        if self.context.clean_mode:
            self.context.cleanup()
