import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Color Palette
C_BG = "#F8F9FA"
C_TEXT = "#2C3E50"
C_MODEL = "#E8F4F8"
C_SAID = "#FFF3E0"
C_BORDER = "#BDC3C7"
C_ARROW = "#7F8C8D"
C_SHADOW = "#000000"

def draw_rounded_box(ax, cx, cy, w, h, text, color, font_size=12, font_weight='normal', text_color=C_TEXT):
    x = cx - w / 2
    y = cy - h / 2
    
    # Shadow
    shadow = patches.FancyBboxPatch((x + 0.05, y - 0.05), w, h, boxstyle="round,pad=0.1,rounding_size=0.2",
                                    linewidth=0, facecolor=C_SHADOW, alpha=0.1)
    ax.add_patch(shadow)
    
    # Box
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.2",
                                 linewidth=1.5, edgecolor=C_BORDER, facecolor=color)
    ax.add_patch(box)
    
    # Text
    ax.text(cx, cy, text, horizontalalignment='center', verticalalignment='center', 
            fontsize=font_size, color=text_color, weight=font_weight, wrap=True, zorder=10)

def draw_arrow(ax, x1, y1, x2, y2, connectionstyle="arc3"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_length=0.4,head_width=0.3", lw=2, color=C_ARROW,
                                connectionstyle=connectionstyle))

def draw_circle(ax, cx, cy, radius, text, color):
    circle = patches.Circle((cx, cy), radius, linewidth=1.5, edgecolor=C_BORDER, facecolor=color, zorder=5)
    ax.add_patch(circle)
    ax.text(cx, cy, text, horizontalalignment='center', verticalalignment='center', 
            fontsize=14, color=C_TEXT, weight='bold', zorder=10)

# SAID Architecture Diagram
fig, ax = plt.subplots(figsize=(8, 7), facecolor=C_BG)
ax.axis('off')
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)

ax.text(4, 7.5, "SAID: Subspace Approximation Projection", fontsize=16, weight='bold', color=C_TEXT, ha='center')

W = 3.5
H = 0.8

# Top to Bottom architecture flow
draw_rounded_box(ax, 4, 6.5, 2.5, H, "Trainable Vector\n$z \\in \\mathbb{R}^d$", C_SAID, font_weight='bold')
draw_arrow(ax, 4, 6.0, 4, 5.5)

draw_rounded_box(ax, 4, 5.0, W, H, "Fastfood Projection Function ($F$)", C_MODEL)
draw_arrow(ax, 4, 4.5, 4, 4.0)

draw_rounded_box(ax, 4, 3.5, 3.5, H, "Full Space Parameter Update\n$\\Delta\\theta \\in \\mathbb{R}^D$", C_SAID, font_weight='bold')
draw_arrow(ax, 4, 3.0, 4, 2.5)

# Original model weight
draw_rounded_box(ax, 1.5, 2.0, 2, H, "Pretrained Weights\n$\\theta_0$", C_MODEL)
draw_circle(ax, 4, 2.0, 0.25, r'$+$', C_SAID)

draw_arrow(ax, 2.6, 2.0, 3.65, 2.0)
draw_arrow(ax, 4, 1.65, 4, 1.1)

draw_rounded_box(ax, 4, 0.6, W, H, "Updated Model Parameters\n$\\theta = \\theta_0 + \\Delta\\theta$", C_MODEL, font_weight='bold')

plt.tight_layout()
plt.savefig('images/said_architecture.png', bbox_inches='tight', dpi=300)
print("Saved images/said_architecture.png")
