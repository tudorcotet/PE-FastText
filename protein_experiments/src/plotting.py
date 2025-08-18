import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Plotting Configuration ---
CUSTOM_PALETTE = ["#414A87", "#CCA53D", "#284C64", "#6E9E5B", "#7F2239"]
sns.set_style("whitegrid")
sns.set_context("talk")
FIG_SIZE = (10, 6)
TITLE_FONT_SIZE = 16
AXIS_FONT_SIZE = 12

def save_plot(fig, output_path: Path, is_catplot=False):
    """Save a Matplotlib figure, handling regular plots and FacetGrids."""
    sns.despine()
    if not is_catplot:
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def annotate_bars(ax, **kwargs):
    """Adds value annotations to bar plots."""
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points',
                    fontsize=10)

def plot_performance_by_task(df: pd.DataFrame, output_dir: Path):
    """Plot model performance (R2 or Accuracy) grouped by task."""
    metric = 'test_r2' if 'test_r2' in df.columns and df['test_r2'].notna().any() else 'test_accuracy'
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.boxplot(data=df, x='task', y=metric, ax=ax)
    
    ax.set_title('Model performance across different tasks', fontsize=TITLE_FONT_SIZE, pad=20)
    ax.set_xlabel('Task', fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=AXIS_FONT_SIZE)
    ax.tick_params(axis='x', rotation=25)
    
    save_plot(fig, output_dir / 'performance_by_task.png')

def plot_pe_vs_baseline(df: pd.DataFrame, output_dir: Path):
    """Compare performance of models with and without positional encoding."""
    metric = 'test_r2' if 'test_r2' in df.columns and df['test_r2'].notna().any() else 'test_accuracy'
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.boxplot(data=df, x='task', y=metric, hue='has_pe', ax=ax)
    
    ax.set_title('Positional encoding vs. baseline performance', fontsize=TITLE_FONT_SIZE, pad=20)
    ax.set_xlabel('Task', fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=AXIS_FONT_SIZE)
    ax.legend(title='Has positional encoding')
    ax.tick_params(axis='x', rotation=25)

    save_plot(fig, output_dir / 'pe_vs_baseline_comparison.png')

def plot_pos_encoder_performance(df: pd.DataFrame, output_dir: Path):
    """Plot performance for each type of positional encoder, faceted by task."""
    pe_df = df[df['pos_encoder'] != 'baseline'].copy()
    metric = 'test_r2' if 'test_r2' in df.columns and pe_df['test_r2'].notna().any() else 'test_accuracy'

    g = sns.catplot(
        data=pe_df, x='pos_encoder', y=metric, col='task',
        kind='bar', col_wrap=3, sharey=False, height=4, aspect=1.2,
        errorbar='sd', palette='viridis'
    )
    g.fig.suptitle('Performance by positional encoder across tasks', fontsize=TITLE_FONT_SIZE, y=1.03)
    g.set_axis_labels('Positional encoder', metric.replace('_', ' ').title())
    g.set_titles("Task: {col_name}")
    
    save_plot(g.fig, output_dir / 'pos_encoder_performance_faceted.png', is_catplot=True)

def plot_tokenization_performance(df: pd.DataFrame, output_dir: Path):
    """Compare performance of k-mer vs. residue tokenization."""
    metric = 'test_r2' if 'test_r2' in df.columns and df['test_r2'].notna().any() else 'test_accuracy'
    ft_df = df[df['embedder_type'] == 'fasttext'].copy()
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.boxplot(data=ft_df, x='task', y=metric, hue='tokenization', ax=ax)
    
    ax.set_title('K-mer vs. residue tokenization performance', fontsize=TITLE_FONT_SIZE, pad=20)
    ax.set_xlabel('Task', fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=AXIS_FONT_SIZE)
    ax.legend(title='Tokenization')
    ax.tick_params(axis='x', rotation=25)

    save_plot(fig, output_dir / 'tokenization_comparison.png')
    
def plot_finetuning_performance(df: pd.DataFrame, output_dir: Path):
    """Compare performance of fine-tuned vs. frozen models."""
    metric = 'test_r2' if 'test_r2' in df.columns and df['test_r2'].notna().any() else 'test_accuracy'
    ft_df = df[df['fine_tune'].notna()].copy()
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.boxplot(data=ft_df, x='task', y=metric, hue='fine_tune', ax=ax)
    
    ax.set_title('Fine-tuning vs. frozen model performance', fontsize=TITLE_FONT_SIZE, pad=20)
    ax.set_xlabel('Task', fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=AXIS_FONT_SIZE)
    ax.legend(title='Fine-tuned')
    ax.tick_params(axis='x', rotation=25)
    
    save_plot(fig, output_dir / 'finetuning_comparison.png')

def plot_predictor_performance(df: pd.DataFrame, output_dir: Path):
    """Compare performance across different predictor models."""
    metric = 'test_r2' if 'test_r2' in df.columns and df['test_r2'].notna().any() else 'test_accuracy'

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.barplot(data=df, x='task', y=metric, hue='predictor_type', ax=ax, errorbar='sd')
    
    ax.set_title('Performance by predictor model', fontsize=TITLE_FONT_SIZE, pad=20)
    ax.set_xlabel('Task', fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel(f'Mean {metric.replace("_", " ").title()}', fontsize=AXIS_FONT_SIZE)
    ax.legend(title='Predictor')
    ax.tick_params(axis='x', rotation=25)

    save_plot(fig, output_dir / 'predictor_performance.png')

def plot_fusion_performance(df: pd.DataFrame, output_dir: Path):
    """Compare 'add' vs. 'concatenate' fusion methods for positional encodings."""
    metric = 'test_r2' if 'test_r2' in df.columns and df['test_r2'].notna().any() else 'test_accuracy'
    pe_df = df[df['pos_encoder'] != 'baseline'].copy()
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.boxplot(data=pe_df, x='pos_encoder', y=metric, hue='fusion', ax=ax)
    
    ax.set_title('Positional encoding fusion method performance', fontsize=TITLE_FONT_SIZE, pad=20)
    ax.set_xlabel('Positional encoder', fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=AXIS_FONT_SIZE)
    ax.legend(title='Fusion method')
    ax.tick_params(axis='x', rotation=25)

    save_plot(fig, output_dir / 'fusion_method_comparison.png')
