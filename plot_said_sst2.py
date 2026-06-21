import matplotlib.pyplot as plt

# Data extracted from the log output (recorded every 20 steps)
step_losses = [
    0.7029, 0.6674, 0.6511, 0.6290, 0.6096, 0.6000, 0.5563, # Epoch 1
    0.5571, 0.5587, 0.5363, 0.5338, 0.5120, 0.5194, 0.5225, # Epoch 2
    0.5089, 0.4983, 0.4857, 0.4694, 0.4850, 0.4945, 0.4753, # Epoch 3
    0.4692, 0.4909, 0.4378, 0.4453, 0.4485, 0.4700, 0.4446, # Epoch 4
    0.4444, 0.4739, 0.4399, 0.4424, 0.4145, 0.4406, 0.4114  # Epoch 5
]

val_accuracy = [78.78, 79.59, 80.16, 80.39, 81.08]

# Calculate global steps (approx 135 steps per epoch based on batch size)
steps_per_epoch = 135
global_steps_loss = []
for epoch in range(5):
    for i in range(7):
        global_steps_loss.append(epoch * steps_per_epoch + i * 20)
        
global_steps_val = [(i + 1) * steps_per_epoch for i in range(5)]

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Global Training Step', fontsize=12)
ax1.set_ylabel('Training Loss', color=color, fontsize=12)
ax1.plot(global_steps_loss, step_losses, color=color, alpha=0.8, linewidth=2, label='Train Loss')
ax1.scatter(global_steps_loss, step_losses, color=color, s=30)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Validation Accuracy (%)', color=color, fontsize=12)
ax2.plot(global_steps_val, val_accuracy, marker='s', markersize=8, color=color, linewidth=2, linestyle='--', label='Val Accuracy')
ax2.tick_params(axis='y', labelcolor=color)

# Add legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

fig.suptitle('SAID DistilBERT Fine-Tuning on SST2', fontsize=16, fontweight='bold')
fig.tight_layout()

plt.savefig('images/said_distilbert_sst2_metrics.png', dpi=300, bbox_inches='tight')
print("Graph saved to images/said_distilbert_sst2_metrics.png")
