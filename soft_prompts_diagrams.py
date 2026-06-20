import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Modern elegant colors
C_BG = "#FFFFFF"
C_TEXT = "#2C3E50"
C_MODEL = "#E8F4F8"
C_ADAPTER = "#FFF3E0"  # For trainable parts
C_ACTION = "#FDEAEA"
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

def draw_fancy_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_length=0.4,head_width=0.3", lw=2, color=C_ARROW,
                                connectionstyle="arc3"))

# ==========================================
# 1. Hard Prompts vs Soft Prompts
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6), facecolor=C_BG)
ax.axis('off')
ax.set_xlim(0, 8)
ax.set_ylim(0, 7)

# Hard Prompt
ax.text(1, 6.5, "Hard Prompt (Discrete Text)", fontsize=14, weight='bold', color=C_TEXT)
draw_rounded_box(ax, 0.5, 5, 3, 1, "Translate to French:\n(Tokens)", "#EAF2D3", font_size=11)
draw_rounded_box(ax, 3.8, 5, 3, 1, "How are you?\n(Input)", C_MODEL, font_size=11)
ax.text(3.65, 5.5, "+", fontsize=16, weight='bold', color=C_TEXT, ha='center', va='center')
draw_fancy_arrow(ax, 3.65, 4.8, 3.65, 3.8)
draw_rounded_box(ax, 2.15, 2.5, 3, 1, "Frozen LLM", C_MODEL, font_size=12, font_weight='bold')

# Soft Prompt
ax.text(1, 1.5, "Soft Prompt (Continuous Vectors)", fontsize=14, weight='bold', color=C_TEXT)
draw_rounded_box(ax, 0.5, 0, 3, 1, "[P1, P2, P3, P4]\n(Trainable Vectors)", C_ADAPTER, font_size=11, font_weight='bold')
draw_rounded_box(ax, 3.8, 0, 3, 1, "How are you?\n(Input)", C_MODEL, font_size=11)
ax.text(3.65, 0.5, "+", fontsize=16, weight='bold', color=C_TEXT, ha='center', va='center')
draw_fancy_arrow(ax, 3.65, 1.8, 3.65, 1.8) # dummy arrow, pointing up to model visually... wait
# Let's adjust soft prompt arrow
draw_fancy_arrow(ax, 3.65, 1.2, 3.65, 2.2)

plt.savefig('prompt_comparison.png', bbox_inches='tight', dpi=300)
plt.close()

# ==========================================
# 2. Validation Loss Plot
# ==========================================
fig, ax = plt.subplots(figsize=(6, 4), facecolor=C_BG)
epochs = [1, 2, 3, 4, 5]
random_loss = [13.80, 10.06, 8.56, 7.25, 5.99]
instruction_loss = [15.81, 14.91, 14.14, 12.78, 11.12]

ax.plot(epochs, random_loss, marker='o', linestyle='-', color='#E74C3C', linewidth=2, label='Random Initialization')
ax.plot(epochs, instruction_loss, marker='s', linestyle='--', color='#3498DB', linewidth=2, label='Instruction Initialization')

ax.set_title('Validation Loss Over Epochs', fontsize=14, weight='bold', color=C_TEXT)
ax.set_xlabel('Epoch', fontsize=12, color=C_TEXT)
ax.set_ylabel('Loss', fontsize=12, color=C_TEXT)
ax.set_xticks(epochs)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('validation_loss_plot.png', bbox_inches='tight', dpi=300)
plt.close()

# ==========================================
# 3. Soft Prompts Architecture
# ==========================================
fig, ax = plt.subplots(figsize=(6, 7), facecolor=C_BG)
ax.axis('off')
ax.set_xlim(0, 6)
ax.set_ylim(0, 7)

draw_rounded_box(ax, 1, 5.5, 4, 1.0, "Trainable Prompt Vectors\n[P_1, P_2, ..., P_m]", C_ADAPTER, font_size=12, font_weight='bold')
draw_rounded_box(ax, 1, 4.0, 4, 1.0, "Frozen Input Embeddings\n[X_1, X_2, ..., X_n]", C_MODEL, font_size=12)

ax.text(3, 3.7, "Concatenate [P; X]", fontsize=12, weight='bold', ha='center', va='center')
draw_fancy_arrow(ax, 3, 3.5, 3, 2.5)

draw_rounded_box(ax, 1, 1.3, 4, 1.0, "Frozen Transformer Layers", C_MODEL, font_size=12, font_weight='bold')
draw_fancy_arrow(ax, 3, 1.1, 3, 0.3)
ax.text(3, 0, "Output Predictions", fontsize=12, weight='bold', ha='center', va='center')

plt.savefig('soft_prompt_architecture.png', bbox_inches='tight', dpi=300)
plt.close()
