import matplotlib.pyplot as plt
import os


# -------------------------
# Paths and Config
# -------------------------

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_FILES = {
    "RNN": "rnn.txt",
    "BLSTM": "blstm.txt",
    "Attention": "attention.txt",
}

# Fill these values manually.
# Use counts for unique_names, novel_names, epochs_trained.
# Use rates in [0, 1] for novelty_rate and diversity.
MANUAL_METRICS = {
    "RNN": {
        "unique_names": 168,
        "novel_names": 147,
        "novelty_rate": 0.735,
        "diversity": 0.84,
        "epochs_trained": 30,
    },
    "BLSTM": {
        "unique_names": 153,
        "novel_names": 118,
        "novelty_rate": 0.59,
        "diversity": 0.765,
        "epochs_trained": 100,
    },
    "Attention": {
        "unique_names": 175,
        "novel_names": 153,
        "novelty_rate": 0.765,
        "diversity": 0.875,
        "epochs_trained": 50,
    },
}


def validate_manual_metrics(metrics):
    required_keys = [
        "unique_names",
        "novel_names",
        "novelty_rate",
        "diversity",
        "epochs_trained",
    ]

    missing = []
    for model in MODEL_FILES:
        model_values = metrics.get(model, {})
        for key in required_keys:
            if key not in model_values or model_values[key] is None:
                missing.append(f"{model}.{key}")

    if missing:
        print("Please fill the following values in MANUAL_METRICS before plotting:")
        for item in missing:
            print(f"- {item}")
        return False

    return True


def save_bar_plot(values, title, ylabel, filename, color="steelblue"):
    labels = list(values.keys())
    scores = list(values.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, scores, color=color)
    plt.title(title)
    plt.xlabel("Models")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close()


def save_grouped_rates_plot(novelty_rate, diversity):
    labels = list(novelty_rate.keys())
    x = list(range(len(labels)))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], [novelty_rate[k] for k in labels], width=width, label="Novelty Rate")
    plt.bar([i + width / 2 for i in x], [diversity[k] for k in labels], width=width, label="Diversity")
    plt.xticks(x, labels)
    plt.title("Novelty Rate and Diversity Comparison")
    plt.xlabel("Models")
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "novelty_diversity_comparison.png"), dpi=150)
    plt.close()


def main():
    if not validate_manual_metrics(MANUAL_METRICS):
        return

    unique_names = {m: MANUAL_METRICS[m]["unique_names"] for m in MODEL_FILES}
    novel_names = {m: MANUAL_METRICS[m]["novel_names"] for m in MODEL_FILES}
    novelty_rate = {m: MANUAL_METRICS[m]["novelty_rate"] for m in MODEL_FILES}
    diversity = {m: MANUAL_METRICS[m]["diversity"] for m in MODEL_FILES}
    epochs_trained = {m: MANUAL_METRICS[m]["epochs_trained"] for m in MODEL_FILES}

    save_bar_plot(unique_names, "Unique Names Comparison", "Unique Count", "unique_names_comparison.png", color="slateblue")
    save_bar_plot(novel_names, "Novel Names Comparison", "Novel Count", "novel_names_comparison.png", color="cadetblue")
    save_bar_plot(novelty_rate, "Novelty Rate Comparison", "Novelty Rate", "novelty_comparison.png", color="teal")
    save_bar_plot(diversity, "Diversity Comparison", "Diversity", "diversity_comparison.png", color="darkorange")
    save_bar_plot(epochs_trained, "Epochs Trained Comparison", "Epoch Count", "epochs_comparison.png", color="mediumpurple")
    save_grouped_rates_plot(novelty_rate, diversity)

    print("Saved manual comparison plots in ./plots folder")
    for model in MODEL_FILES:
        m = MANUAL_METRICS[model]
        print(
            f"{model}: unique={m['unique_names']}, novel={m['novel_names']}, "
            f"novelty={m['novelty_rate']}, diversity={m['diversity']}, epochs={m['epochs_trained']}"
        )


if __name__ == "__main__":
    main()