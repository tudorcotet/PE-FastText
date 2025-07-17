import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from adjustText import adjust_text

# --- Plotting Configuration ---
sns.set_theme(style="ticks", palette="viridis")
FIG_SIZE = (18, 10)
TITLE_FONT_SIZE = 16
AXIS_FONT_SIZE = 12

def plot_umap_comparison(
    embeds_dict: dict,
    labels: list,
    output_path: Path,
    main_title: str,
):
    """
    Creates and saves a UMAP comparison plot for multiple models.
    """
    n_models = len(embeds_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]  # Make it iterable

    unique_labels = sorted(list(set(labels)))
    palette = sns.color_palette("viridis", len(unique_labels))
    color_map = {label: color for label, color in zip(unique_labels, palette)}

    fig.suptitle(main_title, fontsize=TITLE_FONT_SIZE, y=1.02)

    for ax, (model_name, umap_embeds) in zip(axes, embeds_dict.items()):
        ax.scatter(umap_embeds[:, 0], umap_embeds[:, 1], c=[color_map[l] for l in labels], s=20, alpha=0.8)
        ax.set_title(model_name.replace("_", " ").title(), fontsize=AXIS_FONT_SIZE)
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)

        # Add annotations for centroids
        texts = []
        for label_name in unique_labels:
            idx = [i for i, l in enumerate(labels) if l == label_name]
            if not idx:
                continue
            centroid = umap_embeds[idx].mean(axis=0)
            texts.append(ax.text(centroid[0], centroid[1], label_name, fontsize=9, ha='center', va='center'))

        if texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='black'))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot to: {output_path}") 