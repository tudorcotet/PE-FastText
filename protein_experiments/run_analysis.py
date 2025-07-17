import pandas as pd
from pathlib import Path
from fpdf import FPDF
import argparse

# Add src to path to allow for local imports
import sys
# Make the path addition robust to where the script is run from
script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir / 'src'))

from plotting import (
    plot_performance_by_task,
    plot_pe_vs_baseline,
    plot_pos_encoder_performance,
    plot_tokenization_performance,
    plot_finetuning_performance,
    plot_predictor_performance,
    plot_fusion_performance,
)

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
    # Drop failed runs
    df.dropna(subset=['test_r2', 'test_accuracy'], how='all', inplace=True)
    df.fillna({'pos_encoder': 'baseline'}, inplace=True)
    df['has_pe'] = df['pos_encoder'] != 'baseline'
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
    plot_pe_vs_baseline(df, FIGURES_DIR)
    plot_pos_encoder_performance(df, FIGURES_DIR)
    plot_fusion_performance(df, FIGURES_DIR)
    plot_tokenization_performance(df, FIGURES_DIR)
    plot_finetuning_performance(df, FIGURES_DIR)
    plot_predictor_performance(df, FIGURES_DIR)
    
    print(f"All plots saved to '{FIGURES_DIR}' directory.")
    
    # --- 3. Generate PDF Report ---
    pdf = PDF()
    
    plots_to_include = [
        (
            FIGURES_DIR / "performance_by_task.png",
            "Overall model performance by task",
            "This plot shows the distribution of performance scores for each prediction task. It helps identify which tasks are more challenging and reveals the overall variance in model performance."
        ),
        (
            FIGURES_DIR / "pe_vs_baseline_comparison.png",
            "Positional encoding vs. baseline performance",
            "This plot compares models with any type of positional encoding against baseline models (without PE). It provides a high-level view of whether positional information is beneficial across different tasks."
        ),
        (
            FIGURES_DIR / "pos_encoder_performance_faceted.png",
            "Performance by positional encoder across tasks",
            "This faceted plot details the mean performance of each positional encoding strategy for each task separately. This allows for a granular view of which encoders are most effective for specific problems."
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
            "This bar chart shows the mean performance for each type of predictor model. It helps identify which machine learning model is most effective on average for each task."
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