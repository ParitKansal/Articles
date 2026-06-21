import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Color Palette
C_BG = "#F8F9FA"
C_TEXT = "#2C3E50"
C_MODEL = "#E8F4F8"
C_IA3 = "#FFF3E0"
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

# (IA)3 Architecture Diagram
fig, ax = plt.subplots(figsize=(10, 8), facecolor=C_BG)
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 9)

ax.text(5, 8.5, "(IA)³ Injection in a Transformer Block", fontsize=16, weight='bold', color=C_TEXT, ha='center')

W = 2.5
H = 0.8

# Inputs
draw_rounded_box(ax, 2.5, 7.5, W, H, "Queries (Q)", C_MODEL)
draw_rounded_box(ax, 5.0, 7.5, W, H, "Keys (K)", C_MODEL)
draw_rounded_box(ax, 7.5, 7.5, W, H, "Values (V)", C_MODEL)

# IA3 Vectors for K and V
draw_rounded_box(ax, 5.0, 6.2, W, H, "Key Vector ($l_k$)", C_IA3, font_weight='bold')
draw_rounded_box(ax, 7.5, 6.2, W, H, "Value Vector ($l_v$)", C_IA3, font_weight='bold')

# Multiply nodes
draw_circle(ax, 5.0, 5.3, 0.25, r'$\odot$', C_IA3)
draw_circle(ax, 7.5, 5.3, 0.25, r'$\odot$', C_IA3)

# Connect Inputs to Multipliers
draw_arrow(ax, 5.0, 7.1, 5.0, 6.6)
draw_arrow(ax, 5.0, 5.8, 5.0, 5.55)

draw_arrow(ax, 7.5, 7.1, 7.5, 6.6)
draw_arrow(ax, 7.5, 5.8, 7.5, 5.55)

# Attention Block
draw_rounded_box(ax, 5.0, 4.0, 6, 1.2, "Multi-Head Attention\n$Softmax(QK^T / \\sqrt{d}) V$", C_MODEL, font_weight='bold')

draw_arrow(ax, 2.5, 7.1, 3.5, 4.6)
draw_arrow(ax, 5.0, 5.05, 5.0, 4.6)
draw_arrow(ax, 7.5, 5.05, 6.5, 4.6)

# FFN Block
draw_rounded_box(ax, 5.0, 2.2, 4, H, "Feed-Forward Network\nIntermediate Activation ($h$)", C_MODEL)
draw_arrow(ax, 5.0, 3.4, 5.0, 2.6)

# IA3 Vector for FFN
draw_rounded_box(ax, 8.2, 2.2, W, H, "FFN Vector ($l_f$)", C_IA3, font_weight='bold')

# Multiply node for FFN
draw_circle(ax, 5.0, 1.1, 0.25, r'$\odot$', C_IA3)

draw_arrow(ax, 5.0, 1.8, 5.0, 1.35)
draw_arrow(ax, 8.2, 1.8, 5.25, 1.1, connectionstyle="angle,angleA=-90,angleB=0,rad=10")

# Output
draw_rounded_box(ax, 5.0, 0.2, 3, H, "Transformer Output", C_MODEL, font_weight='bold')
draw_arrow(ax, 5.0, 0.85, 5.0, 0.6)

plt.tight_layout()
plt.savefig('images/ia3_architecture.png', bbox_inches='tight', dpi=300)
print("Saved images/ia3_architecture.png")
