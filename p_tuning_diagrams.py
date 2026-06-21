import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

# 1. P-Tuning Prompt Encoder
fig, ax = plt.subplots(figsize=(8, 6), facecolor=C_BG)
ax.axis('off')
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)

ax.text(4, 7.5, "P-Tuning: Prompt Encoder Architecture", fontsize=14, weight='bold', color=C_TEXT, ha='center')

draw_rounded_box(ax, 2.5, 6.0, 3, 0.8, "Virtual Prompt IDs", C_MODEL)
draw_fancy_arrow(ax, 4.0, 5.87, 4.0, 4.93)

draw_rounded_box(ax, 2.5, 4.0, 3, 0.8, "Embedding Layer", C_PREFIX)
draw_fancy_arrow(ax, 4.0, 3.87, 4.0, 2.93)

draw_rounded_box(ax, 2.5, 2.0, 3, 0.8, "Bi-LSTM + MLP", C_PREFIX, font_weight='bold')
draw_fancy_arrow(ax, 4.0, 1.87, 4.0, 0.93)

draw_rounded_box(ax, 1.5, 0.0, 5, 0.8, "Continuous Prompt Embeddings ($P_i$)", C_PREFIX, font_weight='bold')

plt.savefig('p_tuning_encoder.png', bbox_inches='tight', dpi=300)
plt.close()

# 2. P-Tuning Full Architecture
fig, ax = plt.subplots(figsize=(11, 7), facecolor=C_BG)
ax.axis('off')
ax.set_xlim(0, 11)
ax.set_ylim(0, 8)

ax.text(5.5, 7.5, "P-Tuning Input Pipeline", fontsize=14, weight='bold', color=C_TEXT, ha='center')

# Inputs
draw_rounded_box(ax, 0.5, 6.0, 4, 0.8, "Virtual Prompt Tokens", C_PREFIX)
draw_rounded_box(ax, 6.5, 6.0, 4, 0.8, "Text Tokens", C_MODEL)

draw_fancy_arrow(ax, 2.5, 5.87, 2.5, 4.93)
draw_fancy_arrow(ax, 8.5, 5.87, 8.5, 4.93)

# Processing
draw_rounded_box(ax, 0.5, 4.0, 4, 0.8, "Prompt Encoder\n(Bi-LSTM + MLP)", C_PREFIX, font_weight='bold')
draw_rounded_box(ax, 6.5, 4.0, 4, 0.8, "Standard LLM\nEmbedding Layer", C_MODEL)

draw_fancy_arrow(ax, 2.5, 3.87, 2.5, 2.93)
draw_fancy_arrow(ax, 8.5, 3.87, 8.5, 2.93)

# Output Embeddings
draw_rounded_box(ax, 0.5, 2.0, 4, 0.8, "Continuous Prompt Embeddings", C_PREFIX)
draw_rounded_box(ax, 6.5, 2.0, 4, 0.8, "Word Embeddings", C_MODEL)

# Combine
draw_fancy_arrow(ax, 2.5, 1.87, 4.5, 0.93)
draw_fancy_arrow(ax, 8.5, 1.87, 6.5, 0.93)

draw_rounded_box(ax, 3.5, 0.0, 4, 0.8, "Concatenate & Pass to LLM", C_MODEL, font_weight='bold')

plt.savefig('p_tuning_architecture.png', bbox_inches='tight', dpi=300)
plt.close()
