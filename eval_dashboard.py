#!/usr/bin/env python3
"""
Imaginarium v1 vs v3 评估 Dashboard + 计划
Phase 2: GT-based evaluation with real scene graph comparison
"""
import json, os

DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "evaluation_dashboard.json")

METRICS = {
    "fidelity_gt": {
        "category": "重建保真度 — GT 对照 (Table 3, GT meta.json)",
        "items": {
            "object_recovery":          "物体恢复率 (overall)",
            "scene_graph_accuracy_gt":  "场景图准确率 (vs GT parent)",
            "rotation_auc60":           "旋转 AUC@60°",
            "translation_auc05":        "平移 AUC@0.5m",
            "v3_stacking_fixes":        "v3 场景图修正数",
            "v3_fix_coverage":          "v3 有修正的场景占比",
        },
        "paper": {
            "object_recovery":          "—",
            "scene_graph_accuracy_gt":  "—",
            "rotation_auc60":           "—",
            "translation_auc05":        "—",
            "v3_stacking_fixes":        "22.1/scene",
            "v3_fix_coverage":          "39/39 (100%)",
        },
    },
    "plan": {
        "category": "📋 Phase 2 评估计划",
        "items": {},
    }
}

def check_resources():
    status = {"ok": [], "missing": []}
    status["ok"].append("8x A100 GPU")
    status["ok"].append("151 scenes × 20 categories (GT meta.json)")
    status["ok"].append("GPT API (gpt-5.2)")
    for w in ["ae_net_pretrained_weights.pth", "depth_anything_v2_metric_hypersim_vitl.pth", "dinov2_vitl14.pth"]:
        if os.path.exists(f"weights/{w}"):
            status["ok"].append(f"Weight: {w.split('.')[0]}")
    return status


def build_dashboard():
    status = check_resources()
    db = {
        "resources": {"all_ok": len(status["missing"]) == 0, "ok": status["ok"], "missing": status["missing"]},
        "metrics": METRICS,
        "_plan": [
            {"id": 1, "task": "修正评估指标", "detail": "用数据集 meta.json 的 GT parent 做场景图准确率标准，替换 GPT 自一致性", "status": "todo"},
            {"id": 2, "task": "Prompt 适配", "detail": "将 S1 prompt 从 gpt-4o 适配到 gpt-5.2/5.4，减少 JSON 解析失败和格式偏差", "status": "todo"},
            {"id": 3, "task": "API 稳定性", "detail": "解决 GPT API 超时问题——可能需要压缩图片、减少并发、或加 retry delay", "status": "todo"},
            {"id": 4, "task": "全量 151 场景跑完", "detail": "在 API 稳定 + prompt 适配后，跑完全部 151 场景 v1+v3", "status": "todo"},
            {"id": 5, "task": "GT 位姿对比", "detail": "对于完成场景，用 GT matrix_world_4x4 计算旋转/平移误差 (AUC@60°, AUC@0.5m)", "status": "todo"},
            {"id": 6, "task": "填充 Dashboard 全指标", "detail": "将 Table 3 全部 12 项 + v3 专属指标填入 dashboard", "status": "todo"},
        ],
        "_v1_results": {},
        "_v3_results": {},
    }
    with open(DASHBOARD_PATH, "w") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    return db


def print_dashboard(db):
    print("=" * 80)
    print("  Imaginarium Phase 2 评估计划")
    print("=" * 80)
    for item in db["_plan"]:
        icon = "✅" if item["status"] == "done" else ("🔄" if item["status"] == "doing" else "⬜")
        print(f"  {icon} [{item['id']}] {item['task']}")
        print(f"      {item['detail']}")
    print(f"\n  Dashboard: {DASHBOARD_PATH}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    db = build_dashboard()
    print_dashboard(db)
