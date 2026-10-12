from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIG
# ============================================================

seeds = [0, 50, 500, 1000, 4000]

experiments = {
    #"Flat terrain": "logs/rsl_rl/bipedo_manager/2026-05-07_16-27-29/play_metrics",
    "500 entornos": "logs/rsl_rl/bipedo_manager/seed_500_{}",
    "1000 entornos": "logs/rsl_rl/bipedo_manager/seed_1000_{}",
    "4000 entornos": "logs/rsl_rl/bipedo_manager/seed_4000_{}",
}

metrics = [
    "Metrics/base_velocity/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw",
    "Metrics/distancia_recorrida",
]

colors = ["tab:blue", "tab:orange", "tab:green"]

save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)

# ============================================================
# STYLE
# ============================================================

scale = 1.3

plt.rcParams.update({
    'font.size': 10 * scale,
    'axes.titlesize': 12 * scale,
    'axes.labelsize': 10 * scale,
    'xtick.labelsize': 9 * scale,
    'ytick.labelsize': 9 * scale,
    'legend.fontsize': 9 * scale,
    'lines.linewidth': 2,
})

# ============================================================
# FUNCTIONS
# ============================================================

def load_scalar(path, metric):

    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()

    scalar = ea.Scalars(metric)

    steps = np.array([x.step for x in scalar])
    values = np.array([x.value for x in scalar])

    return steps, values


def smooth(y, box_pts=10):

    box = np.ones(box_pts) / box_pts

    return np.convolve(y, box, mode='same')

# ============================================================
# LOOP OVER METRICS
# ============================================================

for metric in metrics:

    plt.figure(figsize=(10, 6))

    for (exp_name, path_template), color in zip(experiments.items(), colors):

        all_values = []

        # ====================================================
        # LOAD ALL SEEDS
        # ====================================================

        for seed in seeds:

            path = path_template.format(seed)

            try:

                steps, values = load_scalar(path, metric)

                all_values.append(values)

            except Exception as e:

                print(f"Error loading:")
                print(path)
                print(metric)
                print(e)

        # ====================================================
        # ALIGN LENGTHS
        # ====================================================

        min_len = min(len(v) for v in all_values)

        all_values = np.array([
            v[:min_len] for v in all_values
        ])

        aligned_steps = steps[:min_len]

        # ====================================================
        # STATS
        # ====================================================

        mean = np.mean(all_values, axis=0)
        std = np.std(all_values, axis=0)

        # OPTIONAL SMOOTHING
        mean = smooth(mean, 10)
        std = smooth(std, 10)

        # ====================================================
        # PLOT
        # ====================================================

        plt.plot(
            aligned_steps,
            mean,
            label=exp_name,
            color=color
        )

        plt.fill_between(
            aligned_steps,
            mean - std,
            mean + std,
            alpha=0.2,
            color=color
        )

    # ========================================================
    # FIGURE CONFIG
    # ========================================================

    metric_name = metric.split("/")[-1]
    if metric_name == "distancia_recorrida":
        metric_name = "Distancia recorrida (m)"
        title_name = "Distancia recorrida"
    if metric_name == "error_vel_xy":
        metric_name = "Error lineal (m/s)"
        title_name = "Error de seguimiento de velocidad lineal"
    if metric_name == "error_vel_yaw":
        metric_name = "Error angular (rad/s)"
        title_name = "Error de seguimiento de velocidad angular"
    plt.title(title_name)
    plt.xlabel("Iteración")
    plt.ylabel(metric_name)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # ========================================================
    # SAVE
    # ========================================================

    save_path = os.path.join(save_dir, f"{title_name}.png")

    plt.savefig(save_path, dpi=300, format="png")
    #plt.show()
    print(f"Saved: {save_path}")

    plt.close()