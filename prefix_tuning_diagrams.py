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
fig, ax = plt.subplots(figsize=(10.5, 6), facecolor=C_BG)
ax.axis('off')
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 8)

# Prompt Tuning (Left)
ax.text(2.0, 7.5, "Prompt Tuning", fontsize=14, weight='bold', color=C_TEXT, ha='center')
draw_rounded_box(ax, 0.5, 6, 1.5, 0.8, "Prompts", C_PREFIX, font_weight='bold')
draw_rounded_box(ax, 2.0, 6, 1.5, 0.8, "Input Tokens", C_MODEL)
draw_fancy_arrow(ax, 2.0, 5.8, 2.0, 4.0)

draw_rounded_box(ax, 0.5, 3.0, 3, 0.8, "Transformer Layer 1", C_MODEL)
draw_fancy_arrow(ax, 2.0, 2.8, 2.0, 2.0)
draw_rounded_box(ax, 0.5, 1.0, 3, 0.8, "Transformer Layer 2", C_MODEL)

# Prefix Tuning (Right)
ax.text(8.0, 7.5, "Prefix Tuning", fontsize=14, weight='bold', color=C_TEXT, ha='center')
draw_rounded_box(ax, 6.5, 6, 3, 0.8, "Input Tokens", C_MODEL)
draw_fancy_arrow(ax, 8.0, 5.8, 8.0, 4.0)

draw_rounded_box(ax, 6.5, 3.0, 3, 0.8, "Transformer Layer 1", C_MODEL)
draw_rounded_box(ax, 4.5, 3.1, 1.2, 0.6, "Prefixes", C_PREFIX, font_size=10, font_weight='bold')
draw_fancy_arrow(ax, 5.8, 3.4, 6.4, 3.4)

draw_fancy_arrow(ax, 8.0, 2.8, 8.0, 2.0)

draw_rounded_box(ax, 6.5, 1.0, 3, 0.8, "Transformer Layer 2", C_MODEL)
draw_rounded_box(ax, 4.5, 1.1, 1.2, 0.6, "Prefixes", C_PREFIX, font_size=10, font_weight='bold')
draw_fancy_arrow(ax, 5.8, 1.4, 6.4, 1.4)

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
draw_fancy_arrow(ax, 3.2, 4.87, 3.2, 3.93)
draw_fancy_arrow(ax, 5.0, 4.87, 5.0, 3.93)
draw_fancy_arrow(ax, 6.9, 4.87, 6.9, 3.93)
draw_fancy_arrow(ax, 8.7, 4.87, 8.7, 3.93)

# Concat Boxes
draw_rounded_box(ax, 2.5, 3, 3.2, 0.8, "Concat: $[P_K; K]$", C_MODEL, font_weight='bold')
draw_rounded_box(ax, 6.2, 3, 3.2, 0.8, "Concat: $[P_V; V]$", C_MODEL, font_weight='bold')

# Arrows to MHA
draw_fancy_arrow(ax, 1.25, 4.87, 1.25, 1.93)
draw_fancy_arrow(ax, 4.1, 2.87, 4.1, 1.93)
draw_fancy_arrow(ax, 7.8, 2.87, 7.8, 1.93)

# MHA Box
draw_rounded_box(ax, 0.5, 1, 8.9, 0.8, "Multi-Head Attention", C_MODEL, font_size=12, font_weight='bold')

plt.savefig('prefix_attention.png', bbox_inches='tight', dpi=300)
plt.close()

# 3. MLP Reparameterization
fig, ax = plt.subplots(figsize=(11, 7), facecolor=C_BG)
ax.axis('off')
ax.set_xlim(0, 11)
ax.set_ylim(0, 7)

# Left: During Training
ax.text(2.75, 6.5, "During Training", fontsize=14, weight='bold', color=C_TEXT, ha='center')

draw_rounded_box(ax, 2.0, 5.2, 1.5, 0.6, "$E_{prefix}$\n(Small)", C_PREFIX, font_weight='bold')
draw_fancy_arrow(ax, 2.75, 5.07, 2.75, 4.53)

draw_rounded_box(ax, 1.75, 3.8, 2.0, 0.6, "MLP Generator\n(Trainable)", C_MODEL, font_weight='bold')
draw_fancy_arrow(ax, 2.75, 3.67, 2.75, 3.13)

draw_rounded_box(ax, 1.5, 2.4, 2.5, 0.6, "Prefix Tensors $P_K, P_V$\n(Large)", C_PREFIX, font_weight='bold')
draw_fancy_arrow(ax, 2.75, 2.27, 2.75, 1.63)

draw_rounded_box(ax, 1.0, 0.5, 3.5, 1.0, "Attention Layers\n(Frozen)", C_MODEL, font_size=12, font_weight='bold')

# Right: During Inference
ax.text(8.25, 6.5, "During Inference", fontsize=14, weight='bold', color=C_TEXT, ha='center')

# Discarded MLP
draw_rounded_box(ax, 7.25, 3.8, 2.0, 0.6, "MLP Generator\n(Discarded)", "#E0E0E0", font_weight='bold', text_color="#A0A0A0")
ax.plot([7.15, 9.35], [3.7, 4.5], color='#E74C3C', lw=3)
ax.plot([7.15, 9.35], [4.5, 3.7], color='#E74C3C', lw=3)

# Saved Prefix Tensors
draw_rounded_box(ax, 7.0, 2.4, 2.5, 0.6, "Saved Tensors $P_K, P_V$\n(Static)", C_PREFIX, font_weight='bold')
draw_fancy_arrow(ax, 8.25, 2.27, 8.25, 1.63)

draw_rounded_box(ax, 6.5, 0.5, 3.5, 1.0, "Attention Layers\n(Frozen)", C_MODEL, font_size=12, font_weight='bold')

plt.savefig('prefix_mlp.png', bbox_inches='tight', dpi=300)
plt.close()
