import re
import matplotlib.pyplot as plt

log_file = "./logs/train.log"

train_loss = []
train_dice = []
val_loss = []
val_dice = []
val_dice_per_class = [[], [], []]

with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i in range(len(lines)):
    line = lines[i]

    # Train
    if "Train Loss:" in line:
        match = re.search(r"Train Loss: ([0-9.]+) \| Dice: ([0-9.]+)", line)
        if match:
            train_loss.append(float(match.group(1)))
            train_dice.append(float(match.group(2)))

    # Val
    if "Val Loss:" in line:
        match = re.search(r"Val Loss:\s+([0-9.]+) \| Dice: ([0-9.]+)", line)
        if match:
            val_loss.append(float(match.group(1)))
            val_dice.append(float(match.group(2)))

    # Dice per class
    if "Val Dice per class:" in line:
        match = re.search(r"Val Dice per class: ([0-9., ]+)", line)
        if match:
            values = [float(x) for x in match.group(1).split(",")]
            for c in range(3):
                val_dice_per_class[c].append(values[c])

epochs = range(1, len(train_loss) + 1)

# ===== Plot =====
plt.figure(figsize=(15, 10))

# Loss
plt.subplot(2, 2, 1)
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Val Loss")
plt.title("Loss")
plt.legend()

# Dice
plt.subplot(2, 2, 2)
plt.plot(epochs, train_dice, label="Train Dice")
plt.plot(epochs, val_dice, label="Val Dice")
plt.title("Dice")
plt.legend()

# Dice per class
plt.subplot(2, 1, 2)
for c in range(3):
    plt.plot(epochs, val_dice_per_class[c], label=f"Class {c}")
plt.title("Val Dice per Class")
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")
plt.show()