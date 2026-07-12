"""Generate 4-step comparison image for Imaginarium v3 pipeline."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os

RESULT = 'saved_results/custom_scene_result'
OUT = os.path.join(RESULT, 'custom_comparison.png')

images = [
    ('S0: Input',                          'demo/custom_scene.png'),
    ('S1: Scene Parsing',                  os.path.join(RESULT, 'S1_scene_parsing_results/scene_graph_final.png')),
    ('S3: Pose Inference',                 os.path.join(RESULT, 'S3_pose_inference/pose_prediction_stitched.png')),
    ('S4: Final Render (v3 stack-aware)',  os.path.join(RESULT, 'S4_layout_refinement/custom_scene_render_simu.png')),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax, (title, path) in zip(axes.flat, images):
    if os.path.exists(path):
        img = Image.open(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, f'Missing: {os.path.basename(path)}', transform=ax.transAxes, ha='center')
    ax.axis('off')

plt.tight_layout()
plt.savefig(OUT, dpi=120, bbox_inches='tight')
print(f'Comparison saved: {OUT}')
