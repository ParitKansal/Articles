import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, w, h, text, color='white', edgecolor='black', font_size=12):
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=edgecolor, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, horizontalalignment='center', verticalalignment='center', fontsize=font_size, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=2, color='black'))

# 1. Traditional Fine-Tuning
fig, ax = plt.subplots(figsize=(4, 5))
ax.axis('off')
ax.set_xlim(0, 4)
ax.set_ylim(0, 5)

draw_box(ax, 0.5, 3.5, 3, 1, "Pretrained Model\n(7B Parameters)", color='#e0f7fa')
draw_arrow(ax, 2, 3.5, 2, 2.5)
draw_box(ax, 0.5, 1.5, 3, 1, "Update Every Parameter", color='#ffcdd2')
draw_arrow(ax, 2, 1.5, 2, 0.5)
draw_box(ax, 0.5, -0.5, 3, 1, "Task-Specific Model", color='#c8e6c9')
plt.savefig('traditional_finetuning.png', bbox_inches='tight', dpi=300)
plt.close()

# 2. Transformer with Adapters
fig, ax = plt.subplots(figsize=(5, 9))
ax.axis('off')
ax.set_xlim(0, 5)
ax.set_ylim(-1, 8)

y = 7
h = 0.6
space = 0.4
w = 3.5
x = 0.75

steps = [
    ("Input", '#f5f5f5'),
    ("Multi-Head Attention", '#e3f2fd'),
    ("Add & Normalize", '#f5f5f5'),
    ("Adapter", '#fff9c4'),
    ("Feed Forward Network", '#e3f2fd'),
    ("Add & Normalize", '#f5f5f5'),
    ("Adapter", '#fff9c4'),
    ("Output", '#f5f5f5')
]

for i, (text, color) in enumerate(steps):
    draw_box(ax, x, y, w, h, text, color=color, font_size=12)
    if i < len(steps) - 1:
        draw_arrow(ax, x + w/2, y, x + w/2, y - space)
    y -= (h + space)

plt.savefig('adapter_transformer.png', bbox_inches='tight', dpi=300)
plt.close()

# 3. Adapter Bottleneck
fig, ax = plt.subplots(figsize=(5, 7))
ax.axis('off')
ax.set_xlim(0, 6)
ax.set_ylim(0, 7)

y = 5.5
w = 3
x = 1.5
h = 0.6
space = 0.6

steps = [
    ("Input (dim = d)", '#f5f5f5'),
    ("Down Projection\n(d → r)", '#ffcc80'),
    ("Non-Linearity", '#ffe082'),
    ("Up Projection\n(r → d)", '#ffcc80'),
    ("Output (dim = d)", '#f5f5f5')
]

for i, (text, color) in enumerate(steps):
    draw_box(ax, x, y, w, h, text, color=color)
    if i < len(steps) - 1:
        draw_arrow(ax, x + w/2, y, x + w/2, y - space)
    y -= (h + space)

# Add residual connection
ax.annotate("", xy=(x + w + 0.2, 0.5 + h/2), xytext=(x + w, 5.5 + h/2),
            arrowprops=dict(arrowstyle="->", connectionstyle="bar,fraction=-0.2", lw=2, color='black'))
ax.text(x + w + 0.6, 3, "Residual Connection", rotation=270, verticalalignment='center', fontsize=11)

plt.savefig('adapter_bottleneck.png', bbox_inches='tight', dpi=300)
plt.close()
