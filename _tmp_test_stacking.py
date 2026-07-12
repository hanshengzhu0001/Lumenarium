import json, sys
sys.path.insert(0, '.')
from modules._s3_legacy_functions import detect_stacking_pairs, _collect_dependents

S1 = 'saved_results/custom_scene_result/S1_scene_parsing_results'
sg = json.load(open(S1 + '/scene_graph_result_final.json'))
pairs = detect_stacking_pairs(S1, sg)
print('=== detected stacking pairs (upper ON lower) ===')
for lo, up in pairs:
    print('  %-22s ON  %s' % (up, lo))
print()
print('=== support-chain propagation (descendants of each upper) ===')
for lo, up in pairs:
    print('  %-22s dependents=%s' % (up, _collect_dependents(up, sg)))
