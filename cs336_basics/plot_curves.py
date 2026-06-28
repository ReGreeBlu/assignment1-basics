import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv("output/tinystories_train_log.csv")
df_valid = pd.read_csv("output/tinystories_valid_log.csv")

# Plot the results
# plt.style.use("classic")
plt.rcParams["font.size"] = 12
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  

# loss vs step
ax1.plot(df_train["step"], df_train["loss"], label="train",
         color="tab:red", linestyle="-", linewidth=3)
ax1.plot(df_valid["step"], df_valid["loss"], label="valid",
         color="tab:blue", linestyle="-", marker="o", alpha=0.6, linewidth=2)
ax1.grid(True, alpha=0.3)
ax1.set_title("Loss vs Step")
ax1.set_xlabel("step"); ax1.set_ylabel("loss"); ax1.legend()

# loss vs time
ax2.plot(df_train["step"], df_train["loss"], label="train",
         color="tab:red", linestyle="-", linewidth=3)
ax2.plot(df_valid["step"], df_valid["loss"], label="valid",
         color="tab:blue", linestyle="-", marker="o", alpha=0.6, linewidth=2)
ax2.grid(True, alpha=0.3)
ax2.set_title("Loss vs Time")
ax2.set_xlabel("time (s)"); ax2.set_ylabel("loss"); ax2.legend()

fig.suptitle("Training Curves", fontsize=18)
plt.tight_layout()
plt.savefig("record/loss_curves.png")

