import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Color Palette
C_BG = "#F8F9FA"
C_TEXT = "#2C3E50"
C_MODEL = "#E8F4F8"
C_PREFIX = "#FFF3E0"
C_BORDER = "#BDC3C7"
C_ARROW = "#7F8C8D"
C_SHADOW = "#000000"

def draw_rounded_box(ax, x, y, w, h, text, color, font_size=12, text_color=C_TEXT, font_weight='normal'):
    shadow = patches.FancyBboxPatch((x+0.05, y-0.05), w, h, boxstyle="round,pad=0.1,rounding_size=0.2",
                                    linewidth=0, facecolor=C_SHADOW, alpha=0.1)
    ax.add_patch(shadow)
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.2",
                                 linewidth=1.5, edgecolor=C_BORDER, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, horizontalalignment='center', verticalalignment='center', 
            fontsize=font_size, color=text_color, weight=font_weight, wrap=True, zorder=10)

def draw_fancy_arrow(ax, x1, y1, x2, y2, connectionstyle="arc3"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_length=0.4,head_width=0.3", lw=2, color=C_ARROW,
                                connectionstyle=connectionstyle))

# 1. Prompt Tuning vs Prefix Tuning
fig, ax = plt.subplots(figsize=(10, 6), facecolor=C_BG)
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)

# Prompt Tuning (Left)
ax.text(2.5, 7.5, "Prompt Tuning", fontsize=14, weight='bold', color=C_TEXT, ha='center')
draw_rounded_box(ax, 1, 6, 3, 0.8, "Input Tokens", C_MODEL)
draw_rounded_box(ax, 1, 5, 3, 0.8, "Trainable Prompts", C_PREFIX, font_weight='bold')
draw_fancy_arrow(ax, 2.5, 4.8, 2.5, 4.0)
draw_rounded_box(ax, 1, 3.0, 3, 0.8, "Transformer Layer 1", C_MODEL)
draw_fancy_arrow(ax, 2.5, 2.8, 2.5, 2.0)
draw_rounded_box(ax, 1, 1.0, 3, 0.8, "Transformer Layer 2", C_MODEL)

# Prefix Tuning (Right)
ax.text(7.5, 7.5, "Prefix Tuning", fontsize=14, weight='bold', color=C_TEXT, ha='center')
draw_rounded_box(ax, 6, 6, 3, 0.8, "Input Tokens", C_MODEL)
draw_fancy_arrow(ax, 7.5, 5.8, 7.5, 4.0)

draw_rounded_box(ax, 5.5, 3.0, 4, 0.8, "Transformer Layer 1", C_MODEL)
draw_rounded_box(ax, 4.3, 3.1, 1, 0.6, "Prefixes", C_PREFIX, font_size=10, font_weight='bold')
draw_fancy_arrow(ax, 5.3, 3.4, 5.5, 3.4)

draw_fancy_arrow(ax, 7.5, 2.8, 7.5, 2.0)

draw_rounded_box(ax, 5.5, 1.0, 4, 0.8, "Transformer Layer 2", C_MODEL)
draw_rounded_box(ax, 4.3, 1.1, 1, 0.6, "Prefixes", C_PREFIX, font_size=10, font_weight='bold')
draw_fancy_arrow(ax, 5.3, 1.4, 5.5, 1.4)

plt.savefig('prefix_vs_prompt.png', bbox_inches='tight', dpi=300)
plt.close()

# 2. Prefix Attention
fig, ax = plt.subplots(figsize=(10, 6), facecolor=C_BG)
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)

ax.text(5, 6.5, "Prefix Tuning in Attention Layer", fontsize=14, weight='bold', color=C_TEXT, ha='center')

# Q, P_K, K, P_V, V
draw_rounded_box(ax, 0.5, 5, 1.5, 0.8, "Query (Q)", C_MODEL)
draw_rounded_box(ax, 2.45, 5, 1.5, 0.8, "$P_K$\n(Trainable)", C_PREFIX, font_weight='bold')
draw_rounded_box(ax, 4.25, 5, 1.5, 0.8, "Key (K)", C_MODEL)
draw_rounded_box(ax, 6.15, 5, 1.5, 0.8, "$P_V$\n(Trainable)", C_PREFIX, font_weight='bold')
draw_rounded_box(ax, 7.95, 5, 1.5, 0.8, "Value (V)", C_MODEL)

# Arrows to Concat
draw_fancy_arrow(ax, 3.2, 5, 3.2, 3.8)
draw_fancy_arrow(ax, 5.0, 5, 5.0, 3.8)
draw_fancy_arrow(ax, 6.9, 5, 6.9, 3.8)
draw_fancy_arrow(ax, 8.7, 5, 8.7, 3.8)

# Concat Boxes
draw_rounded_box(ax, 2.5, 3, 3.2, 0.8, "Concat: $[P_K; K]$", C_MODEL, font_weight='bold')
draw_rounded_box(ax, 6.2, 3, 3.2, 0.8, "Concat: $[P_V; V]$", C_MODEL, font_weight='bold')

# Arrows to MHA
draw_fancy_arrow(ax, 1.25, 5, 1.25, 1.8)
draw_fancy_arrow(ax, 4.1, 3, 4.1, 1.8)
draw_fancy_arrow(ax, 7.8, 3, 7.8, 1.8)

# MHA Box
draw_rounded_box(ax, 0.5, 1, 8.9, 0.8, "Multi-Head Attention", C_MODEL, font_size=12, font_weight='bold')

plt.savefig('prefix_attention.png', bbox_inches='tight', dpi=300)
plt.close()
