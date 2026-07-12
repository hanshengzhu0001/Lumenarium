"""Generate 6-panel full pipeline comparison for custom_scene3 v3."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os

RESULT = 'saved_results/custom_scene3_result'
S4 = os.path.join(RESULT, 'S4_layout_refinement')
OUT = os.path.join(RESULT, 'custom_scene3_comparison.png')

panels = [
    # (title, path)
    ('S0: Input',                      'demo/custom_scene3.png'),
    ('S1: 2D Detection',               os.path.join(RESULT, 'S1_scene_parsing_results/scene_graph_final.png')),
    ('S1: 3D Layout (render_s1)',      os.path.join(S4, 'custom_scene3_render_s1.png')),
    ('S3: Pose Inference',             os.path.join(RESULT, 'S3_pose_inference/pose_prediction_stitched.png')),
    ('S3: Pre-sim (render_s3)',        os.path.join(S4, 'custom_scene3_render_s3.png')),
    ('S4: Final Render (v3 stack)',    os.path.join(S4, 'custom_scene3_render_simu.png')),
]

fig, axes = plt.subplots(2, 3, figsize=(24, 13))
for ax, (title, path) in zip(axes.flat, panels):
    if os.path.exists(path):
        img = Image.open(path)
        if img.mode == 'RGBA':
            img = img.convert('RGBA')
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        ax.imshow(img)
        ax.set_title(title, fontsize=13, fontweight='bold')
    else:
        ax.text(0.5, 0.5, f'Missing:\n{os.path.basename(path)}',
                transform=ax.transAxes, ha='center', fontsize=10)
    ax.axis('off')

plt.tight_layout(pad=1.5)
plt.savefig(OUT, dpi=110, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUT}')
