import pandas as pd  # type: ignore[import]
from pathlib import Path
from fpdf import FPDF  # type: ignore[import]
import argparse

# Add src to path to allow for local imports
import sys
# Make the path addition robust to where the script is run from
script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir / 'src'))

# Local plotting utilities from companion module (no stubs)
from plotting import (  # type: ignore[import]
    plot_performance_by_task,
    plot_tokenization_performance,
    plot_fusion_performance,
    CUSTOM_PALETTE,
    save_plot,
    annotate_bars,
)


import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore[import]
import numpy as np  # type: ignore
from scipy import stats  # type: ignore[import]
from typing import List, Tuple
from functools import reduce
from pathlib import Path
from itertools import combinations

sns.set_theme(style="ticks", palette="viridis")


def _metric_column(df: pd.DataFrame) -> str:
    """Return the preferred metric column present in df."""
    if 'test_r2' in df.columns and df['test_r2'].notna().any():
        return 'test_r2'
    return 'test_accuracy'


def plot_bar_per_pos_encoder(df: pd.DataFrame, output_dir: Path):
    """Bar plot with mean ± SE per positional encoder (incl. baseline & ESM2), grouped by task."""
    metric = _metric_column(df)

    # Map every row to a display encoder category and define colors
    def _display_enc(row):
        if row['embedder_type'] == 'esm2':
            return 'ESM2'
        pe = row['pos_encoder']
        if pe == 'baseline': return 'None'
        if pe == 'sinusoid': return 'Sinusoidal'
        if pe == 'rope': return 'RoPE'
        if pe == 'ft_alibi': return 'AliBi'
        return pe # Fallback

    df = df.copy()
    df['Method'] = df.apply(_display_enc, axis=1)
    
    color_map = {
        "None": "#414A87", 
        "Sinusoidal": "#CCA53D", 
        "RoPE": "#284C64", 
        "AliBi": "#6E9E5B", 
        "ESM2": "#7F2239"
    }
    
    order = ["None", "Sinusoidal", "RoPE", "AliBi", "ESM2"]

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=df,
        x='task',
        y=metric,
        hue='Method',
        hue_order=order,
        ax=ax,
        palette=color_map,
        errorbar=('ci', 95),  # Use 95% confidence interval for error bars
        capsize=0.1
    )
    
    annotate_bars(ax)
    
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_xlabel('Dataset')
    ax.set_title('Performance by Dataset')
    ax.tick_params(axis='x', rotation=15)
    ax.legend(title='Method', bbox_to_anchor=(1.01, 1), loc='upper left')

    output_path = output_dir / 'performance_by_dataset_bars.png'
    save_plot(fig, output_path)


def _annotate_pvalue(ax, x1: float, x2: float, y: float, label: str):
    """Draws a p-value bracket between two boxes."""
    ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y], lw=1.3, c='k')
    ax.text((x1 + x2) / 2, y + 0.012, label, ha='center', va='bottom', size=9)


def plot_composite_boxes(df: pd.DataFrame, output_dir: Path):
    """Composite figure with three box plots + t-test annotations."""
    metric = _metric_column(df)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    # 1) Positional encoding vs baseline
    sns.boxplot(data=df, x='has_pe', y=metric, ax=axes[0], palette=CUSTOM_PALETTE[:2], hue='has_pe', legend=False)
    axes[0].set_xlabel('Has positional encoding')
    axes[0].set_title('Positional encoding vs baseline')

    groups = [df[df['has_pe'] == val][metric].dropna() for val in [False, True]]
    if all(len(g) > 1 for g in groups):
        stat, pval = stats.ttest_ind(*groups, equal_var=False)
        _annotate_pvalue(axes[0], 0, 1, df[metric].max(), f'p={pval:.3g}')

    # 2) Fusion method performance (only rows with PE)
    pe_df = df[df['pos_encoder'] != 'baseline']
    sns.boxplot(data=pe_df, x='fusion', y=metric, ax=axes[1], palette=CUSTOM_PALETTE[:2], hue='fusion', legend=False)
    axes[1].set_xlabel('Fusion')
    axes[1].set_title('Fusion method')

    fus_groups = [pe_df[pe_df['fusion'] == f][metric].dropna() for f in pe_df['fusion'].unique()]
    if len(fus_groups) == 2 and all(len(g) > 1 for g in fus_groups):
        stat, pval = stats.ttest_ind(*fus_groups, equal_var=False)
        _annotate_pvalue(axes[1], 0, 1, pe_df[metric].max(), f'p={pval:.3g}')

    # 3) Tokenization performance (fasttext only)
    ft_df = df[df['embedder_type'] == 'fasttext']
    sns.boxplot(data=ft_df, x='tokenization', y=metric, ax=axes[2], palette=CUSTOM_PALETTE[:2], hue='tokenization', legend=False)
    axes[2].set_xlabel('Tokenization')
    axes[2].set_title('Tokenization comparison')

    tok_groups = [ft_df[ft_df['tokenization'] == t][metric].dropna() for t in ft_df['tokenization'].unique()]
    if len(tok_groups) == 2 and all(len(g) > 1 for g in tok_groups):
        stat, pval = stats.ttest_ind(*tok_groups, equal_var=False)
        _annotate_pvalue(axes[2], 0, 1, ft_df[metric].max(), f'p={pval:.3g}')

    for ax in axes:
        ax.set_ylabel(metric.replace('_', ' ').title())

    fig.tight_layout()
    output_path = output_dir / 'analysis_composite.png'
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_predictor_performance(df: pd.DataFrame, output_dir: Path):
    """Bar plot of performance by predictor model split by embedder type (fasttext vs esm2)."""
    metric = _metric_column(df)
    df = df.copy()
    df['embedder_group'] = df['embedder_type'].replace({'fasttext': 'FastText', 'esm2': 'ESM2'})
    
    color_map = {"FastText": CUSTOM_PALETTE[0], "ESM2": CUSTOM_PALETTE[4]}

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=df,
        x='predictor_type',
        y=metric,
        hue='embedder_group',
        ax=ax,
        errorbar='se',
        palette=color_map,
    )
    
    annotate_bars(ax)
    
    ax.set_xlabel('Predictor')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title('Performance by Predictor Model')
    ax.legend(title='Embedder')
    output_path = output_dir / 'predictor_performance.png'
    save_plot(fig, output_path)


# ---------- Task-split box plots ----------

def plot_2x2_grid_with_stats(df: pd.DataFrame, x: str, y: str, base_order: List[str], base_box_pairs: List[Tuple[str, str]], output_path: Path, title: str):
    """Creates a 2x2 grid of box plots for the 4 main tasks with statistical annotations."""
    from statannotations.Annotator import Annotator
    
    tasks = ['beta_lactamase_complete', 'fluorescence', 'mpp', 'stability']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=False)
    axes = axes.flatten()

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_df = df[df['task'] == task]
        
        # Dynamically determine order and pairs based on available data for the task
        order = [cat for cat in base_order if cat in task_df[x].unique()]
        box_pairs = [pair for pair in base_box_pairs if all(cat in order for cat in pair)]

        # Slice palette to match the number of categories to avoid warnings
        palette = CUSTOM_PALETTE[:len(order)]
        
        plot_params = {
            'data': task_df,
            'x': x,
            'y': y,
            'order': order,
            'ax': ax,
            'palette': palette,
            'hue': x,
            'legend': False
        }
        
        sns.boxplot(**plot_params)
        
        # Only add annotations if there are pairs to annotate
        if box_pairs:
            plot_params_for_annotator = plot_params.copy()
            del plot_params_for_annotator['ax']
            
            annotator = Annotator(ax, box_pairs, **plot_params_for_annotator)
            annotator.configure(test='t-test_ind', text_format='simple', loc='inside', verbose=0)
            annotator.apply_and_annotate()
        
        ax.set_title(f"task = {task}")
        ax.set_xlabel('')
        ax.set_ylabel(y.replace('_', ' ').title())

    # Create a single legend for the whole figure using the base order
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in CUSTOM_PALETTE[:len(base_order)]]
    fig.legend(handles, base_order, loc='center right', bbox_to_anchor=(1.1, 0.5), title=x.replace('_', ' ').title())

    fig.suptitle(title, fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    save_plot(fig, output_path)

def plot_pe_vs_baseline_task(df: pd.DataFrame, output_dir: Path):
    metric = _metric_column(df)
    df = df.copy()
    df['embedder_group'] = np.select(
        [df['embedder_type'] == 'esm2', df['pos_encoder'] == 'baseline'],
        ['esm2', 'fasttext_baseline'],
        default='fasttext_pe',
    )
    order = ['fasttext_baseline', 'fasttext_pe', 'esm2']
    pairs = [('fasttext_baseline', 'fasttext_pe'), ('fasttext_pe', 'esm2')]
    plot_2x2_grid_with_stats(df, 'embedder_group', metric, order, pairs, output_dir / 'pe_vs_baseline_grid.png', 'Baseline vs PE vs ESM2')

def plot_fusion_task(df: pd.DataFrame, output_dir: Path):
    metric = _metric_column(df)
    pe_df = df[df['pos_encoder'] != 'baseline'].copy()
    pe_df['fusion_group'] = pe_df['fusion']
    esm_df = df[df['embedder_type'] == 'esm2'].copy()
    esm_df['fusion_group'] = 'esm2'
    combined = pd.concat([pe_df, esm_df], ignore_index=True)
    order = ['add', 'concatenate', 'esm2']
    pairs = [('add', 'concatenate')]
    plot_2x2_grid_with_stats(combined, 'fusion_group', metric, order, pairs, output_dir / 'fusion_grid.png', 'Fusion method vs ESM2')

def plot_tokenization_task(df: pd.DataFrame, output_dir: Path):
    metric = _metric_column(df)
    ft_df = df[df['embedder_type'] == 'fasttext'].copy()
    esm_df = df[df['embedder_type'] == 'esm2'].copy()
    esm_df['tokenization'] = 'esm2'
    combined = pd.concat([ft_df, esm_df], ignore_index=True)
    order = ['k-mer', 'residue', 'esm2']
    pairs = [('k-mer', 'residue')]
    plot_2x2_grid_with_stats(combined, 'tokenization', metric, order, pairs, output_dir / 'tokenization_grid.png', 'Tokenization vs ESM2')


# --- Configuration ---
FIGURES_DIR = Path("./figures")
REPORT_FILE = "protein_experiments_report.pdf"

# --- PDF Report Class ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Protein Experiments Analysis Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_plot(self, image_path, title, interpretation):
        if not Path(image_path).exists():
            print(f"Warning: Plot not found at {image_path}, skipping.")
            return
        self.add_page()
        self.chapter_title(title)
        self.image(image_path, x=10, y=30, w=190)
        self.set_y(30 + 125) # Position after the image
        self.chapter_title("Interpretation")
        self.chapter_body(interpretation)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the results DataFrame."""
    df = df.copy()  # Make an explicit copy to avoid SettingWithCopyWarning

    # Drop failed runs
    df.dropna(subset=['test_r2', 'test_accuracy'], how='all', inplace=True)

    # --- Filter out specific PEs ---
    unwanted_pes = ['learned', 'alibi']
    df = df[~df['pos_encoder'].isin(unwanted_pes)]

    # Fill NA and create new columns
    df['pos_encoder'] = df['pos_encoder'].fillna('baseline')
    df['has_pe'] = df['pos_encoder'] != 'baseline'

    # --- Remove DeepLoc & SSP tasks ---
    df = df[~df['task'].str.contains('deeploc', case=False, na=False)]
    df = df[~df['task'].str.contains('ssp', case=False, na=False)]
    return df

def main(data_file: Path):
    """Main analysis script."""
    # --- 1. Load and Preprocess Data ---
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        return
        
    FIGURES_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(data_file)
    df = preprocess_data(df)
    
    # --- 2. Generate Plots ---
    plot_performance_by_task(df, FIGURES_DIR)
    plot_bar_per_pos_encoder(df, FIGURES_DIR)
    plot_composite_boxes(df, FIGURES_DIR)
    plot_fusion_performance(df, FIGURES_DIR)  # kept for reference
    plot_tokenization_performance(df, FIGURES_DIR)
    plot_predictor_performance(df, FIGURES_DIR)
    plot_pe_vs_baseline_task(df, FIGURES_DIR)
    plot_fusion_task(df, FIGURES_DIR)
    plot_tokenization_task(df, FIGURES_DIR)
    
    print(f"All plots saved to '{FIGURES_DIR}' directory.")
    
    # --- 3. Generate PDF Report ---
    pdf = PDF()
    
    # Re-ordered / updated figures in the report
    plots_to_include = [
        (
            FIGURES_DIR / "pos_encoder_bars.png",
            "Performance by positional encoder",
            "Mean performance with standard error for each positional encoder, including baseline FastText and ESM2, across all datasets."
        ),
        (
            FIGURES_DIR / "analysis_composite.png",
            "Key comparative analyses",
            "Left to right: baseline vs positional encoding, fusion method comparison, and tokenization effect. p-values from two-sample t-tests are annotated above each pair of boxes."
        ),
        (
            FIGURES_DIR / "fusion_method_comparison.png",
            "Positional encoding fusion method performance",
            "This plot compares the two fusion methods ('add' vs. 'concatenate') for combining token and positional embeddings. It helps determine which method is generally more effective across different encoders."
        ),
        (
            FIGURES_DIR / "tokenization_comparison.png",
            "K-mer vs. residue tokenization performance",
            "This plot compares the two tokenization strategies for FastText models. It helps determine whether a residue-level or k-mer-based approach is more suitable for each task."
        ),
        (
            FIGURES_DIR / "finetuning_comparison.png",
            "Fine-tuning vs. frozen model performance",
            "This plot shows the impact of fine-tuning the pre-trained embedder on downstream task performance. It indicates whether task-specific adaptation of the embedding layer is beneficial."
        ),
        (
            FIGURES_DIR / "predictor_performance.png",
            "Performance by predictor model",
            "Bar plot with standard errors for each predictor model, including experiments that use ESM2 embeddings."
        ),
        (
            FIGURES_DIR / "pe_vs_baseline_task.png",
            "Baseline vs PE vs ESM2",
            "Box plots comparing the performance of FastText models with and without positional encoding, and ESM2 embeddings."
        ),
        (
            FIGURES_DIR / "fusion_task.png",
            "Fusion method vs ESM2",
            "Box plots comparing the performance of different fusion methods (add, concatenate) and ESM2 embeddings."
        ),
        (
            FIGURES_DIR / "tokenization_task.png",
            "Tokenization vs ESM2",
            "Box plots comparing the performance of different tokenization strategies (k-mer, residue) and ESM2 embeddings."
        )
    ]

    for path, title, interpretation in plots_to_include:
        # Ensure the path is a string for the fpdf library
        pdf.add_plot(str(path), title, interpretation)
            
    pdf.output(REPORT_FILE, 'F')
    print(f"\nAnalysis report saved to '{REPORT_FILE}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis on protein experiment results.")
    parser.add_argument(
        '--data_file',
        type=Path,
        default=Path("data/results_protein_experiments.parquet"),
        help='Path to the Parquet file with experiment results.'
    )
    args = parser.parse_args()
    main(args.data_file) 