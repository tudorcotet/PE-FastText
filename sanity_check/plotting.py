import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# --- Plotting configuration ---
CUSTOM_PALETTE = ["#414A87", "#CCA53D", "#284C64", "#6E9E5B", "#7F2239"]
sns.set_style("whitegrid")
sns.set_context("talk")

def plot_umap_comparison(embeds_dict: dict, labels: np.ndarray, output_path: Path, main_title: str):
    """Generate and save a UMAP comparison plot."""
    pos_encoders = list(embeds_dict.keys())
    n_plots = len(pos_encoders)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 6, 5), sharex=True, sharey=True)
    if n_plots == 1:
        axes = [axes] # Ensure axes is always a list
        
    unique_labels = sorted(list(set(labels)))
    colors = sns.color_palette(CUSTOM_PALETTE, n_colors=len(unique_labels))
    color_map = {label: color for label, color in zip(unique_labels, colors)}

    for i, pe_name in enumerate(pos_encoders):
        ax = axes[i]
        embeds_2d = embeds_dict[pe_name]
        
        # Standardize display names
        display_name = pe_name.replace('_', ' ')
        if 'rope' in pe_name.lower(): display_name = 'RoPE'
        if 'ft alibi' in display_name.lower(): display_name = 'AliBi'
        
        for label in unique_labels:
            mask = np.array(labels) == label
            ax.scatter(embeds_2d[mask, 0], embeds_2d[mask, 1], label=label, alpha=0.7, color=color_map[label], s=15)
        
        ax.set_title(display_name.title())
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.05, 0.5), title="Class")
    
    fig.suptitle(main_title, fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig) 