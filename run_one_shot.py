#!/usr/bin/env python3
"""
Imaginarium 一步到位脚本
用法: python run_one_shot.py <input_image_path> [--gpu GPU_ID] [--no-clean]
"""

import sys
import os
import json
import time
import subprocess
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

REPO = Path(__file__).parent.resolve()
DEMO_DIR = REPO / "demo"
RESULTS_DIR = REPO / "saved_results"


def setup_env(gpu_id: int):
    """设置环境变量"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = "/ssd/kevinzyz/huggingface"
    blender_path = str(REPO / "third_party" / "blender-4.3.2-linux-x64")
    os.environ["PATH"] = os.environ["PATH"] + ":" + blender_path
    return blender_path


def copy_input(src: Path) -> Path:
    """复制输入图到 demo/，返回目标路径"""
    DEMO_DIR.mkdir(exist_ok=True)
    stem = src.stem
    dst = DEMO_DIR / f"{stem}.png"
    if src.resolve() != dst.resolve():
        import shutil
        shutil.copy2(src, dst)
    # 验证 RGB
    im = Image.open(dst).convert("RGB")
    im.save(dst)
    print(f"  ✓ 输入图已就绪: {dst} ({im.size})")
    return dst


def run_pipeline(image_path: Path, clean: bool):
    """运行 pipeline，返回 (success, result_dir)"""
    cmd = [
        "python", "run_imaginarium_I2Layout.py",
        str(image_path),
    ]
    if clean:
        cmd.append("--clean")

    env = os.environ.copy()
    # 激活 conda 环境
    conda_prefix = "/ssd/kevinzyz/miniconda3"
    activate = f"source {conda_prefix}/etc/profile.d/conda.sh && conda activate imaginarium"
    full_cmd = f"{activate} && {' '.join(cmd)}"

    log_path = REPO.parent / "dl_logs" / f"{image_path.stem}.log"
    log_path.parent.mkdir(exist_ok=True)

    print(f"  ▶ 启动 pipeline: {' '.join(cmd)}")
    print(f"  日志: {log_path}")

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            ["bash", "-c", full_cmd],
            cwd=str(REPO),
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

    # 监控 pipeline.log
    result_name = image_path.stem + "_result"
    pipeline_log = RESULTS_DIR / result_name / "pipeline.log"
    stages = {}
    last_progress = time.time()

    while proc.poll() is None:
        if pipeline_log.exists():
            try:
                lines = pipeline_log.read_text(encoding="utf-8", errors="ignore").splitlines()
                for line in lines:
                    if "开始阶段:" in line or "结束阶段:" in line:
                        key = line.split("开始阶段:" if "开始阶段:" in line else "结束阶段:")[1].strip()
                        ts = line[:19]
                        stages[ts] = key
            except:
                pass

        # 每 120s 打印一次进度
        if time.time() - last_progress > 120:
            alive = "RUNNING" if proc.poll() is None else "STOPPED"
            latest = ""
            if pipeline_log.exists():
                lines = pipeline_log.read_text(errors="ignore").splitlines()
                for l in reversed(lines[-20:]):
                    if any(k in l for k in ["开始阶段", "结束阶段", "ERROR", "GPT Time", "正在执行"]):
                        latest = l[11:].strip()[:80]
                        break
            print(f"  … [{alive}] {latest}")
            last_progress = time.time()

        time.sleep(5)

    rc = proc.returncode
    print(f"  process exited: rc={rc}")

    # 检查是否成功
    success = False
    if pipeline_log.exists():
        content = pipeline_log.read_text(errors="ignore")
        if "Completed Successfully" in content or "PIPELINE COMPLETED" in content:
            success = True

    return success, RESULTS_DIR / result_name, log_path


def make_comparison(result_dir: Path, input_path: Path):
    """生成 INPUT → S1 → S3 → SIMU 四连对比图"""
    try:
        inp = Image.open(input_path).convert("RGB").resize((512, 512))
        s1 = Image.open(result_dir / "S4_layout_refinement" / f"{input_path.stem}_render_s1.png").convert("RGB").resize((512, 512))
        s3 = Image.open(result_dir / "S4_layout_refinement" / f"{input_path.stem}_render_s3.png").convert("RGB").resize((512, 512))
        simu = Image.open(result_dir / "S4_layout_refinement" / f"{input_path.stem}_render_simu.png").convert("RGB").resize((512, 512))
    except Exception as e:
        print(f"  ⚠ 无法生成对比图: {e}")
        return None

    W, H = 512, 512
    gap = 10
    canvas = Image.new("RGB", (4 * W + 3 * gap, H + 40), (255, 255, 255))
    canvas.paste(inp, (0, 40))
    canvas.paste(s1, (W + gap, 40))
    canvas.paste(s3, (2 * W + 2 * gap, 40))
    canvas.paste(simu, (3 * W + 3 * gap, 40))
    d = ImageDraw.Draw(canvas)
    for i, txt in enumerate(["INPUT", "S1 Layout", "S3 Pre-Sim", "SIMU Final"]):
        x = i * (W + gap) + W // 2 - 40
        d.text((x, 10), txt, fill=(0, 0, 0))

    out_path = result_dir / f"{input_path.stem}_comparison.png"
    canvas.save(out_path)
    print(f"  ✓ 对比图已生成: {out_path}")
    return out_path


def print_summary(result_dir: Path):
    """打印结果摘要"""
    pipeline_log = result_dir / "pipeline.log"
    if not pipeline_log.exists():
        return

    import re, datetime
    lines = pipeline_log.read_text(encoding="utf-8", errors="ignore").splitlines()

    def parse_time(s):
        m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", s)
        if m:
            return datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        return None

    evts = []
    for l in lines:
        for k in ["开始阶段: S0", "结束阶段: S0", "结束阶段: S1", "结束阶段: S2", "结束阶段: S3", "Layout Optimization Done", "Cleanup complete"]:
            if k in l:
                t = parse_time(l)
                if t:
                    evts.append((k, t))
                    break

    if evts:
        print("\n📊 阶段耗时:")
        start = evts[0][1]
        prev = start
        label = {
            "开始阶段: S0": "START",
            "结束阶段: S0": "S0(几何)",
            "结束阶段: S1": "S1(解析)",
            "结束阶段: S2": "S2(检索)",
            "结束阶段: S3": "S3(姿态)",
            "Layout Optimization Done": "S4(布局优化)",
            "Cleanup complete": "完成",
        }
        for k, tt in evts:
            if k == "开始阶段: S0":
                continue
            dur = int((tt - prev).total_seconds())
            cum = int((tt - start).total_seconds())
            print(f"  {label.get(k, k):16s} +{dur:5d}s  (累计 {cum:5d}s)")
            prev = tt

    # 物体数
    sg = result_dir / "S1_scene_parsing_results" / "scene_graph_result.json"
    if sg.exists():
        try:
            d = json.load(open(sg))
            objs = [k for k, v in d.items() if isinstance(v, dict) and "isOnFloor" in v]
            print(f"\n📦 场景物体数: {len(objs)}")
        except:
            pass

    # 输出文件
    s4 = result_dir / "S4_layout_refinement"
    if s4.exists():
        pngs = list(s4.glob("*.png"))
        print(f"📁 输出文件 ({len(pngs)} 张 PNG):")
        for p in sorted(pngs):
            print(f"  {p.name}")


def main():
    parser = argparse.ArgumentParser(description="Imaginarium 一步到位")
    parser.add_argument("input_image", help="输入图片路径")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (default: 0)")
    parser.add_argument("--no-clean", action="store_true", help="不清除中间结果（断点续跑）")
    args = parser.parse_args()

    src = Path(args.input_image).resolve()
    if not src.exists():
        print(f"❌ 文件不存在: {src}")
        sys.exit(1)

    print(f"🚀 Imaginarium 一步到位")
    print(f"  输入: {src}")
    print(f"  GPU: {args.gpu}")

    # Step 1: 环境
    print("\n[Step 1/5] 设置环境...")
    setup_env(args.gpu)
    print(f"  ✓ CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"  ✓ HF_HOME={os.environ['HF_HOME']}")

    # Step 2: 复制输入
    print("\n[Step 2/5] 准备输入图...")
    dst = copy_input(src)
    image_path = dst

    # Step 3: 运行 pipeline
    print("\n[Step 3/5] 运行 pipeline (S0→S4)...")
    t0 = time.time()
    success, result_dir, log_path = run_pipeline(dst, clean=not args.no_clean)
    elapsed = time.time() - t0

    # Step 4: 生成对比图
    print("\n[Step 4/5] 生成对比图...")
    comp_path = make_comparison(result_dir, dst)

    # Step 5: 摘要
    print("\n[Step 5/5] 结果摘要")
    print(f"  总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  结果目录: {result_dir}")
    if comp_path:
        print(f"  对比图: {comp_path}")

    if result_dir.exists():
        print_summary(result_dir)

    if success:
        print("\n✅ Pipeline 成功完成!")
    else:
        print(f"\n⚠ Pipeline 可能有问题，请检查日志: {log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
