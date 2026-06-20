import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Modern elegant colors
C_BG = "#FFFFFF"
C_TEXT = "#2C3E50"
C_MODEL = "#E8F4F8"
C_ADAPTER = "#FFF3E0"
C_ACTION = "#FDEAEA"
C_BORDER = "#BDC3C7"
C_ARROW = "#7F8C8D"
C_SHADOW = "#000000"

def draw_rounded_box(ax, x, y, w, h, text, color, font_size=12, text_color=C_TEXT, font_weight='normal'):
    # Shadow
    shadow = patches.FancyBboxPatch((x+0.05, y-0.05), w, h, boxstyle="round,pad=0.1,rounding_size=0.2",
                                    linewidth=0, facecolor=C_SHADOW, alpha=0.1)
    ax.add_patch(shadow)
    
    # Box
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.2",
                                 linewidth=1.5, edgecolor=C_BORDER, facecolor=color)
    ax.add_patch(box)
    
    # Text
    ax.text(x + w/2, y + h/2, text, horizontalalignment='center', verticalalignment='center', 
            fontsize=font_size, color=text_color, weight=font_weight, wrap=True, zorder=10)

def draw_fancy_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_length=0.4,head_width=0.3", lw=2, color=C_ARROW,
                                connectionstyle="arc3"))

# ==========================================
# 1. Traditional Fine-Tuning
# ==========================================
fig, ax = plt.subplots(figsize=(6, 5), facecolor=C_BG)
ax.axis('off')
ax.set_xlim(0, 6)
ax.set_ylim(-1, 6)

draw_rounded_box(ax, 1, 4.5, 4, 1.0, "Pretrained Model\n(7B Parameters)", C_MODEL, font_size=13, font_weight='bold')
draw_fancy_arrow(ax, 3, 4.3, 3, 3.2)
draw_rounded_box(ax, 1, 2.0, 4, 1.0, "Update Every Parameter\n(Slow & Expensive)", C_ACTION, font_size=13, font_weight='bold')
draw_fancy_arrow(ax, 3, 1.8, 3, 0.7)
draw_rounded_box(ax, 1, -0.5, 4, 1.0, "Task-Specific Model", "#EAF2D3", font_size=13, font_weight='bold')

plt.savefig('traditional_finetuning.png', bbox_inches='tight', dpi=300)
plt.close()

# ==========================================
# 2. Transformer with Adapters
# ==========================================
fig, ax = plt.subplots(figsize=(6, 9), facecolor=C_BG)
ax.axis('off')
ax.set_xlim(0, 6)
ax.set_ylim(-1, 9)

y = 8
h = 0.6
space = 0.6
w = 4
x = 1

steps = [
    ("Input", C_MODEL, 'normal'),
    ("Multi-Head Attention", C_MODEL, 'bold'),
    ("Add & Normalize", C_MODEL, 'normal'),
    ("Adapter Module\n(Trainable)", C_ADAPTER, 'bold'),
    ("Feed Forward Network", C_MODEL, 'bold'),
    ("Add & Normalize", C_MODEL, 'normal'),
    ("Adapter Module\n(Trainable)", C_ADAPTER, 'bold'),
    ("Output", C_MODEL, 'normal')
]

for i, (text, color, weight) in enumerate(steps):
    draw_rounded_box(ax, x, y, w, h, text, color, font_size=12, font_weight=weight)
    if i < len(steps) - 1:
        draw_fancy_arrow(ax, x + w/2, y - 0.05, x + w/2, y - space + 0.05)
    y -= (h + space)

plt.savefig('adapter_transformer.png', bbox_inches='tight', dpi=300)
plt.close()

# ==========================================
# 3. Adapter Bottleneck
# ==========================================
fig, ax = plt.subplots(figsize=(7, 8), facecolor=C_BG)
ax.axis('off')
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)

y = 6.5
w = 3.5
x = 2.25
h = 0.7
space = 0.6

steps = [
    ("Input\n(Dimension = d)", C_MODEL),
    ("Down Projection\n(d → r)", C_ADAPTER),
    ("Non-Linearity\n(e.g., ReLU, GELU)", C_ADAPTER),
    ("Up Projection\n(r → d)", C_ADAPTER),
    ("Output\n(Dimension = d)", C_MODEL)
]

for i, (text, color) in enumerate(steps):
    draw_rounded_box(ax, x, y, w, h, text, color, font_size=12, font_weight='bold')
    if i < len(steps) - 1:
        draw_fancy_arrow(ax, x + w/2, y - 0.05, x + w/2, y - space + 0.05)
    y -= (h + space)

# Add residual connection
ax.annotate("", xy=(x + w + 0.1, 1.3 + h/2), xytext=(x + w + 0.1, 6.5 + h/2),
            arrowprops=dict(arrowstyle="->,head_length=0.4,head_width=0.3", 
                            connectionstyle="bar,fraction=-0.25", lw=2, color=C_ARROW))
ax.text(x + w + 1.7, 3.9, "Residual Connection", rotation=270, verticalalignment='center', fontsize=12, color=C_ARROW, weight='bold')

plt.savefig('adapter_bottleneck.png', bbox_inches='tight', dpi=300)
plt.close()
