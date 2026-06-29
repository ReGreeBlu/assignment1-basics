import subprocess
import os

learning_rates = [
    # 1e-4, 3e-4, 5e-4, 7e-4,
    # 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3,
    # 5e-3, 8e-3, 1e-2, 2e-2, 3e-2, 1e-1, 1, 10
    1e-4, 3e-4, 1e-3, 3e-3, 8e-3, 1e-2, 1e-1, 1, 10
]
num_steps = 500
max_l2_norm = 10
log_path = "output/sweep_lr/no_grad_clip"

base_cmd = [
    "uv", "run", "python3", "-m", "cs336_basics.training_together"
]

for lr in learning_rates:
    cmd = base_cmd + [
        "--max_l2_norm", str(max_l2_norm),
        "--max_learning_rate", str(lr),
        "--num_steps", str(num_steps),
        "--log_path", log_path
    ]

    print(f"========== Running lr={lr} ==========")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"lr={lr} failed with code {result.returncode}")
        continue

    filename_old = os.path.join(log_path, f"tinystories_train_log.csv")
    filename_new = os.path.join(log_path, f"sweep_train_{lr}.csv")
    os.rename(filename_old, filename_new)

    filename_old = os.path.join(log_path, f"tinystories_valid_log.csv")
    filename_new = os.path.join(log_path, f"sweep_valid_{lr}.csv")
    os.rename(filename_old, filename_new)
