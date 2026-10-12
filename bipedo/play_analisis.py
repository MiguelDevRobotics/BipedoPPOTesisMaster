from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIG
# ============================================================

experiments = {
    "Flat terrain": "logs/rsl_rl/bipedo_manager/seed_4000_4000/play_metrics",
    "Uneven terrain": "logs/rsl_rl/bipedo_manager/seed_4000_0/play_metrics",
}

metrics = [
    "Play/error_vel_xy",
    "Play/error_vel_yaw",
    "Play/Distancia_Recorrida",
]

colors = ["tab:blue", "tab:orange", "tab:green"]

MAX_POINTS = 101  # Only 100 episodes

save_dir = "plots_raw"
os.makedirs(save_dir, exist_ok=True)

# ============================================================
# LOAD FUNCTION
# ============================================================

def load_scalar(path, metric):

    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()

    scalar = ea.Scalars(metric)

    steps = np.array([x.step for x in scalar])
    values = np.array([x.value for x in scalar])

    return steps, values

# ============================================================
# MAIN LOOP
# ============================================================

for exp_name, path in experiments.items():

    print("\n" + "="*60)
    print(f"Experiment: {exp_name}")
    print("="*60)

    plt.figure(figsize=(10, 6))
    scale = 1.5
    plt.rcParams.update({
        'font.size': 10 * scale,
        'axes.titlesize': 12 * scale,
        'axes.labelsize': 10 * scale,
        'xtick.labelsize': 9 * scale,
        'ytick.labelsize': 9 * scale,
        'legend.fontsize': 10 * scale,
        'lines.linewidth': 1.5 * scale,
    })

    for metric, color in zip(metrics, colors):

        steps, values = load_scalar(path, metric)

        # ====================================================
        # TRUNCAR A 100 EPISODIOS
        # ====================================================

        steps = steps[:MAX_POINTS]
        values = values[:MAX_POINTS]

        mean_value = np.mean(values)

        print(f"{metric.split('/')[-1]:25s} -> mean (first {MAX_POINTS}): {mean_value:.4f}  Max:{np.max(values)}")

        metric_name = metric.split('/')[-1]

        if metric_name == "Distancia_Recorrida":
            metric_name = "Distance traveled"
        if metric_name == "error_vel_xy":
            metric_name = "Linear error"
        if metric_name == "error_vel_yaw":
            metric_name = "Angular error"
        plt.plot(
            steps,
            values,
            label=f"{metric_name} (mean={mean_value:.2f})",
            color=color
        )

    # ========================================================
    # STYLE
    # ========================================================

    plt.title(f"{exp_name}")
    plt.xlabel("Episodes")
    plt.ylabel("Value")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # ========================================================
    # SAVE
    # ========================================================

    filename = exp_name.lower().replace(" ", "_")
    save_path = os.path.join(save_dir, f"{filename}_first{MAX_POINTS}.png")

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\nSaved: {save_path}")
