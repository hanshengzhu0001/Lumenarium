#!/usr/bin/env python3
"""
批量评估 v1 vs v3。4 GPU 并行，每 GPU 一个场景，按序处理。

用法:
  python batch_eval.py [--limit N] [--dry-run]
  python batch_eval.py --run-name fixdecode --scenes diningroom_01,livingroom_18
"""
import os, sys, json, shutil, subprocess, time, glob
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

PROJECT_ROOT = Path(__file__).parent.absolute()
DATASET_DIR = PROJECT_ROOT / "asset_data/imaginarium_3d_scene_layout_dataset"
DEMO_DIR = PROJECT_ROOT / "demo"
MAX_GPU = 4  # 4 GPU parallel retry for remaining evaluation gaps
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"

# ─── 准备场景 ───────────────────────────────────────────────
def list_scenes():
    scenes = []
    for cat_dir in sorted(DATASET_DIR.iterdir()):
        if not cat_dir.is_dir(): continue
        for scene_dir in sorted(cat_dir.iterdir()):
            if not scene_dir.is_dir(): continue
            png = scene_dir / f"{scene_dir.name}.png"
            meta = scene_dir / f"{scene_dir.name}_meta.json"
            if png.exists() and meta.exists():
                scenes.append((scene_dir.name, str(png), str(meta)))
    return scenes

def pre_convert(scenes):
    """提前将所有 PNG 转 RGB，避免重复 IO"""
    converted = 0
    for name, png_path, _ in scenes:
        for v in ["v1", "v3"]:
            tag = f"{name}_{v}"
            demo_path = DEMO_DIR / f"{tag}.png"
            if demo_path.exists(): continue
            img = Image.open(png_path)
            if img.mode == 'RGBA':
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            else:
                img.save(demo_path); converted += 1; continue
            demo_path.parent.mkdir(exist_ok=True)
            img.save(demo_path)
            converted += 1
    print(f"Pre-converted {converted} images")

# ─── 运行单个场景 ───────────────────────────────────────────
def ensure_mnt_backed_output_root(output_root: Path):
    """Put large eval outputs on /mnt when possible, mirroring saved_results."""
    link_path = output_root if output_root.is_absolute() else PROJECT_ROOT / output_root
    if link_path.exists():
        return
    if output_root.is_absolute():
        link_path.mkdir(parents=True, exist_ok=True)
        return

    mnt_base = Path("/mnt/kevinzyz/artifacts/Imaginarium-repo")
    if mnt_base.exists():
        target = mnt_base / output_root.name
        target.mkdir(parents=True, exist_ok=True)
        link_path.symlink_to(target)
    else:
        link_path.mkdir(parents=True, exist_ok=True)


def make_run_config(config_template: Path, output_root: Path, run_name: str) -> Path:
    if not run_name:
        return config_template

    config_template = config_template.resolve()
    run_config = PROJECT_ROOT / "config" / f"config_{run_name}.yaml"
    content = config_template.read_text(encoding="utf-8")
    replacements = [
        ('save_parent_folder: "saved_results"', f'save_parent_folder: "{output_root.as_posix()}"'),
        ("save_parent_folder: 'saved_results'", f"save_parent_folder: '{output_root.as_posix()}'"),
        ("save_parent_folder: saved_results", f"save_parent_folder: {output_root.as_posix()}"),
    ]
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new, 1)
            break
    else:
        raise RuntimeError(f"Could not rewrite save_parent_folder in {config_template}")

    weight_cache_dir = os.environ.get("IMAGINARIUM_WEIGHT_CACHE_DIR", "").strip()
    if weight_cache_dir:
        weight_cache = Path(weight_cache_dir)
        weight_replacements = {
            "weights/depth_anything_v2_metric_hypersim_vitl.pth": weight_cache / "depth_anything_v2_metric_hypersim_vitl.pth",
            "weights/dinov2_vitl14.pth": weight_cache / "dinov2_vitl14.pth",
            "weights/ae_net_pretrained_weights.pth": weight_cache / "ae_net_pretrained_weights.pth",
        }
        for original, cached in weight_replacements.items():
            if cached.exists():
                content = content.replace(original, cached.as_posix())

    run_config.write_text(content, encoding="utf-8")
    return run_config


def run_one_scene(args):
    scene_name, png_path, meta_path, variant, gpu_id, output_root, config_path, timeout, gpt_max_wait, gpt_max_retries, clean = args
    output_root = Path(output_root)
    tag = f"{scene_name}_{variant}"
    demo_path = DEMO_DIR / f"{tag}.png"
    output_dir = output_root / f"{tag}_result"

    # 已有结果就跳过
    s4_file = output_dir / "S4_layout_refinement"
    if s4_file.exists():
        s4_jsons = glob.glob(str(s4_file / "*_placement_info_s4.json"))
        if s4_jsons:
            return {"scene": scene_name, "variant": variant, "status": "cached", "output": str(output_dir)}

    # 清理旧结果；resume probe 可以保留 S0 等缓存
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if gpt_max_wait:
        env["IMAGINARIUM_GPT_MAX_WAIT"] = str(gpt_max_wait)
    if gpt_max_retries:
        env["IMAGINARIUM_GPT_MAX_RETRIES"] = str(gpt_max_retries)

    if variant == "v3":
        env["IMAGINARIUM_FLOOR_VERIFY_V2"] = "1"
        env["IMAGINARIUM_S3_STACK_AWARE"] = "1"
        env["IMAGINARIUM_S4_STACK_AWARE"] = "1"
        env["IMAGINARIUM_S1_LOWCAT_PASS"] = "1"  # targeted re-detection for low-recall categories
        env["IMAGINARIUM_USE_SAM3_DETECTION"] = "1"  # SAM3 text-based detection replaces GD
        script = str(PROJECT_ROOT / "run_imaginarium_I2Layout_v3.py")
    else:
        env["IMAGINARIUM_USE_SAM3_DETECTION"] = "1"  # also use SAM3 for v1
        script = str(PROJECT_ROOT / "run_imaginarium_I2Layout.py")

    cmd = [sys.executable, script, str(demo_path), "--config", str(config_path)]
    if clean:
        cmd.insert(3, "--clean")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
                               env=env, timeout=timeout)
        elapsed = time.time() - t0
        s4_jsons = glob.glob(str(output_dir / "S4_layout_refinement" / "*_placement_info_s4.json"))
        ok = result.returncode == 0 and bool(s4_jsons)
        if ok:
            log_path = output_dir / "pipeline.log"
            with open(log_path, "a") as f:
                f.write(f"\n[variant={variant} gpu={gpu_id} elapsed={elapsed:.0f}s]\n")
        stderr = result.stderr[-200:] if result.stderr else ""
        if result.returncode == 0 and not s4_jsons:
            stderr = (stderr + "\nmissing S4 placement output").strip()
        return {
            "scene": scene_name, "variant": variant, "status": "ok" if ok else "fail",
            "output": str(output_dir), "elapsed": elapsed,
            "stderr": stderr if not ok else ""
        }
    except subprocess.TimeoutExpired:
        return {"scene": scene_name, "variant": variant, "status": "timeout", "output": str(output_dir)}
    except Exception as e:
        return {"scene": scene_name, "variant": variant, "status": "error", "error": str(e)}
    except Exception as e:
        return {"scene": scene_name, "variant": variant, "status": "error", "error": str(e)}

# ─── 主流程 ─────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--v1-only", action="store_true")
    parser.add_argument("--v3-only", action="store_true")
    parser.add_argument("--run-name", default="", help="Isolated run name, e.g. fixdecode -> saved_results_fixdecode")
    parser.add_argument("--output-root", default="", help="Override output root. Defaults to saved_results or saved_results_<run-name>")
    parser.add_argument("--config-template", default=str(DEFAULT_CONFIG))
    parser.add_argument("--gpu-count", type=int, default=MAX_GPU)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--gpt-max-wait", type=int, default=0, help="Per GPT task max wait seconds; 0 keeps llm_api default")
    parser.add_argument("--gpt-max-retries", type=int, default=0, help="Per GPT call retries; 0 keeps llm_api default")
    parser.add_argument("--scenes", default="", help="Comma-separated scene ids or a text file with one scene id per line")
    parser.add_argument("--no-clean", action="store_true", help="Resume existing outputs instead of deleting each scene result folder")
    args = parser.parse_args()

    scenes = list_scenes()
    if args.scenes:
        scene_filter_path = Path(args.scenes)
        if scene_filter_path.exists():
            selected = {line.strip() for line in scene_filter_path.read_text().splitlines() if line.strip()}
        else:
            selected = {item.strip() for item in args.scenes.split(",") if item.strip()}
        scenes = [s for s in scenes if s[0] in selected]
    if args.limit: scenes = scenes[:args.limit]

    output_root = Path(args.output_root) if args.output_root else Path(
        "saved_results" if not args.run_name else f"saved_results_{args.run_name}"
    )
    ensure_mnt_backed_output_root(output_root)
    config_path = make_run_config(Path(args.config_template), output_root, args.run_name)

    gpu_count = max(1, args.gpu_count)
    gpu_ids_env = os.environ.get("IMAGINARIUM_GPU_IDS", "").strip()
    if gpu_ids_env:
        gpu_ids = [item.strip() for item in gpu_ids_env.split(",") if item.strip()]
        if not gpu_ids:
            raise RuntimeError("IMAGINARIUM_GPU_IDS was set but no GPU ids were parsed")
        gpu_count = min(gpu_count, len(gpu_ids))
        gpu_ids = gpu_ids[:gpu_count]
    else:
        gpu_ids = [str(i) for i in range(gpu_count)]
    print(f"共 {len(scenes)} 场景")
    print(f"run_name: {args.run_name or 'default'}")
    print(f"output_root: {output_root}")
    print(f"config: {config_path}")
    print(f"gpu_ids: {','.join(gpu_ids)}")

    if args.dry_run:
        for name, _, _ in scenes[:10]: print(f"  {name}")
        return

    # 预转换
    pre_convert(scenes)

    # 构建任务队列
    variants = []
    if not args.v3_only: variants.append("v1")
    if not args.v1_only: variants.append("v3")

    tasks = []
    for scene_name, png_path, meta_path in scenes:
        for v in variants:
            # 检查缓存
            tag = f"{scene_name}_{v}"
            out = output_root / f"{tag}_result"
            s4 = out / "S4_layout_refinement"
            if s4.exists() and glob.glob(str(s4 / "*_placement_info_s4.json")):
                continue
            tasks.append((
                scene_name,
                png_path,
                meta_path,
                v,
                gpu_ids[len(tasks) % gpu_count],
                str(output_root),
                str(config_path),
                args.timeout,
                args.gpt_max_wait,
                args.gpt_max_retries,
                not args.no_clean,
            ))

    print(f"待处理: {len(tasks)} 任务, 使用 {gpu_count} GPU 并行")

    if not tasks:
        print("全部完成!")
        return

    # 4 GPU 并行
    results = []
    with ProcessPoolExecutor(max_workers=gpu_count) as pool:
        futures = {pool.submit(run_one_scene, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            status = "✅" if r["status"] == "ok" else ("⏭️" if r["status"] == "cached" else "❌")
            print(f"{status} [{r['variant']}] {r['scene']} ({r['status']})")

    # 汇总
    ok = sum(1 for r in results if r["status"] in ("ok", "cached"))
    fail = sum(1 for r in results if r["status"] == "fail")
    print(f"\n完成: {ok}, 失败: {fail}")
    
    # 输出 JSON 供 dashboard 使用
    result_name = "batch_results.json" if not args.run_name else f"batch_results_{args.run_name}.json"
    with open(PROJECT_ROOT / result_name, "w") as f:
        json.dump(results, f, indent=2)
    print(f"结果已写入 {result_name}")

if __name__ == "__main__":
    main()
