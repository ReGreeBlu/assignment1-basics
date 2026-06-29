import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

log_path = "output/sweep_lr"

# Plot sweep_lr
learning_rates = [
    # 1e-4, 3e-4, 5e-4, 7e-4,
    # 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3,
    # 5e-3, 8e-3, 1e-2, 2e-2, 3e-2, 1e-1, 1, 10,
    1e-4, 1e-3, 4e-3, 1e-2, 1e-1, 1
]

plt.figure(figsize=(8, 6))
for lr in learning_rates:
    src = os.path.join(log_path, f"sweep_train_{lr}.csv")
    df_train = pd.read_csv(src)
    plt.plot(df_train["step"], df_train["loss"], label=f"lr={lr}")
plt.yscale("log")
plt.xlim(left=-50, right=1900)
plt.ylim(bottom=1.7, top=30)
plt.xlabel("step", fontsize=14)
plt.ylabel("loss (log)", fontsize=14)
plt.title("Training Loss Across Learning Rates", fontsize=16)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig("local/sweep_lr_loss_curves.png", dpi=600, bbox_inches="tight", pad_inches=0.3)

# Plot final loss vs lr
src = os.path.join(log_path, f"sweep_valid_*.csv")
pairs = []
for path in glob.glob(src):
    lr = float(path.split("_")[-1][:-4])
    df_valid = pd.read_csv(path)
    final_step = df_valid["step"].iloc[-1]
    if final_step == 1800:
        final_loss = df_valid["loss"].iloc[-1]
    else:
        final_loss = df_valid["loss"].iloc[-2]
    pairs.append((lr, final_loss))
pairs.sort()                               
lrs = [p[0] for p in pairs]
final_losses = [p[1] for p in pairs]

plt.figure(figsize=(8, 8))
plt.plot(lrs, final_losses, marker='o', lw=3)
plt.xscale("log"); plt.yscale("log")
plt.xlim(left=1e-4, right=10)
plt.ylim(bottom=1.5, top=7)
plt.xlabel("learning rate (log)", fontsize=14)
plt.ylabel("final loss (log)", fontsize=14)

best_i = final_losses.index(min(final_losses))
best_lr, best_loss = lrs[best_i], final_losses[best_i]
plt.scatter([best_lr], [best_loss], color="red", zorder=5, s=50)
plt.annotate(f"best:\nlr={best_lr:.1e}\nloss={best_loss:.2f}",
             xy=(best_lr, best_loss),
             xytext=(best_lr, best_loss*1.3),
             arrowprops=dict(arrowstyle="->", lw=2),
             fontsize=14,
             fontweight="bold",
             ha="center")

div_x = 1e-2
plt.axvline(x=div_x, linestyle="--", color="gray", alpha=0.7)
plt.annotate(f"edge of stability",
             xy=(div_x, 3.5),
             xytext=(div_x*0.5, 3.5),
             arrowprops=dict(arrowstyle="->"),
             fontsize=14,
             ha="right",
             va="center")
plt.axvspan(1e-4, 1e-2, alpha=0.1, color="green")
plt.axvspan(1e-2, 10, alpha=0.1, color="red")

plt.title("Validation Loss vs. Learning Rate", fontsize=16)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig("local/sweep_lr_loss_vs_lr.png", dpi=300, bbox_inches="tight", pad_inches=0.3)

