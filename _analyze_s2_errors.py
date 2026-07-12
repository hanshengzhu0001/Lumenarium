import json, re, pandas as pd
from pathlib import Path
from collections import defaultdict, Counter

def tokens(s):
    parts = re.split(r'[^a-z0-9]+', str(s or '').lower())
    stop = {'a','an','the','sm','nn','packed','2k','01','02','03','0','1','2','3','4','5','6','001','002','003','04','05','06'}
    return {p for p in parts if p and p not in stop and not p.isdigit()}

# Load asset info
csv_path = Path('asset_data/imaginarium_asset_info.csv')
df = pd.read_csv(csv_path)
asset_class_map = {}
for _, row in df.iterrows():
    asset_class_map[row['name_en']] = str(row['class_en']).lower().replace('-', '_').replace(' ', '_')

out = Path('saved_results')
true_errors = defaultdict(list)

for scene_dir in sorted(out.glob('*_v3_result')):
    scene_name = scene_dir.name.replace('_v3_result', '')
    s2_path = scene_dir / 'S2_3d_retrieval_results/retrieval_results_final.json'
    if not s2_path.exists(): continue
    s2 = json.loads(s2_path.read_text())

    for pred_id, candidates in s2.items():
        if pred_id.startswith(('floor_', 'wall_', 'ceiling_')): continue
        if not candidates: continue

        top_asset = candidates[0][0]
        pred_class = re.sub(r'(_\d+)+$', '', pred_id.replace('_0_0', ''))
        asset_class = asset_class_map.get(top_asset, '')

        pred_tokens = tokens(pred_class)
        asset_tokens = tokens(top_asset)
        asset_class_tokens = tokens(asset_class)

        # Semantic match check
        semantic_ok = bool(pred_tokens & asset_tokens) or bool(pred_tokens & asset_class_tokens)
        if not semantic_ok:
            for t in pred_tokens:
                for a in (asset_tokens | asset_class_tokens):
                    if len(t) > 3 and len(a) > 3 and (t in a or a in t):
                        semantic_ok = True
                        break
                if semantic_ok: break

        if not semantic_ok:
            true_errors[pred_class].append((top_asset, asset_class, scene_name))

print(f'Total scenes: {len(list(out.glob("*_v3_result")))}\n')
print(f'{"Pred Class":<35} {"Err#":>4} {"Top Retrieved Asset (asset class)":<55} {"Scenes":>20}')
print('-' * 120)
for cls, items in sorted(true_errors.items(), key=lambda x: len(x[1]), reverse=True):
    asset_counts = Counter((a, ac) for a, ac, _ in items).most_common(3)
    scene_set = {s for _, _, s in items}
    examples = ', '.join([f'{a}' for (a, ac), n in asset_counts])
    print(f'{cls:<35} {len(items):>4}  {examples:<55} {",".join(sorted(scene_set)[:5])[:20]:>20}')

# Cross-category matrix
print('\n=== CROSS-CATEGORY PATTERNS ===')
cross = Counter()
for cls, items in true_errors.items():
    for asset, asset_class, scene in items:
        pair = f'{cls} -> {asset_class or "unknown"}'
        cross[pair] += 1
for pair, n in cross.most_common(30):
    print(f'  {pair}: {n}')

# Summary by pred class family
print('\n=== SUM BY PRED CLASS GROUP ===')
groups = Counter()
for cls, items in true_errors.items():
    # Simple grouping by first token
    group = cls.split('_')[0]
    groups[group] += len(items)
for g, n in groups.most_common(20):
    print(f'  {g}: {n}')
