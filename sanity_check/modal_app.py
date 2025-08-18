import modal
from pathlib import Path
from fpdf import FPDF
import sys

# Define the Modal app
app = modal.App("sanity-checks")

# Define paths to local directories to be mounted
src_path = Path(__file__).parent.parent / "src"
sanity_check_path = Path(__file__).parent

# Define the image with all necessary dependencies
image = modal.Image.debian_slim(python_version="3.9").pip_install(
    "gensim>=4.0.0",
    "numpy>=1.20.0",
    "scikit-learn>=1.0.0",
    "matplotlib>=3.0.0",
    "umap-learn>=0.5.0",
    "tqdm>=4.62.0",
    "datasets>=2.0.0",
    "fpdf2>=2.7.0",
    "adjustText",
)

# Define volumes for models and results
model_volume = modal.Volume.from_name("pe-fasttext-models", create_if_missing=True)
dataset_volume = modal.Volume.from_name("pe-fasttext-datasets", create_if_missing=True)
results_volume = modal.Volume.from_name("pe-fasttext-results6", create_if_missing=True)

# Define mounts for the project directories
mounts = [
    modal.Mount.from_local_dir(src_path, remote_path="/root/src"),
    modal.Mount.from_local_dir(sanity_check_path, remote_path="/root/sanity_check"),
]

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Protein Sanity Checks Report', 0, 1, 'C')

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
        self.set_y(30 + 125)
        self.chapter_title("Interpretation")
        self.chapter_body(interpretation)

@app.function(
    image=image,
    volumes={"/datasets": dataset_volume, "/results": results_volume},
    timeout=1800,
)
def calculate_uniref50_stats():
    """Calculates and saves summary statistics for the UniRef50 dataset from the FASTA file."""
    import pandas as pd
    from collections import Counter
    import time
    from tqdm import tqdm

    print("--- Starting UniRef50 statistics calculation from FASTA file ---")
    start_time = time.time()
    
    fasta_path = Path("/datasets/uniref50.fasta")
    if not fasta_path.exists():
        raise FileNotFoundError(f"UniRef50 FASTA file not found at {fasta_path}. Please ensure it has been downloaded to the 'pe-fasttext-datasets' volume.")

    total_sequences = 0
    total_residues = 0
    amino_acid_counts = Counter()
    
    # Get total file size for tqdm progress bar
    total_size = fasta_path.stat().st_size

    # Iterate through the FASTA file to calculate stats
    with open(fasta_path, 'r') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Processing UniRef50 FASTA") as pbar:
            current_seq = []
            for line in f:
                pbar.update(len(line.encode('utf-8')))
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequence = "".join(current_seq)
                        total_sequences += 1
                        total_residues += len(sequence)
                        amino_acid_counts.update(sequence)
                        current_seq = []
                else:
                    current_seq.append(line)
            # Process the last sequence in the file
            if current_seq:
                sequence = "".join(current_seq)
                total_sequences += 1
                total_residues += len(sequence)
                amino_acid_counts.update(sequence)

    # Final calculations
    avg_seq_length = total_residues / total_sequences if total_sequences > 0 else 0
    
    # Create a DataFrame for the summary
    summary_data = {
        "Statistic": [
            "Total number of sequences",
            "Total number of amino acids (residues)",
            "Average sequence length",
        ],
        "Value": [
            f"{total_sequences:,}",
            f"{total_residues:,}",
            f"{avg_seq_length:.2f}",
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Create a DataFrame for amino acid distribution
    aa_df = pd.DataFrame.from_dict(amino_acid_counts, orient='index', columns=['Count'])
    aa_df = aa_df.sort_index()
    
    # Save to CSV files
    output_dir = Path("/results/dataset_stats")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "uniref50_summary_stats.csv"
    distribution_path = output_dir / "uniref50_amino_acid_distribution.csv"
    
    summary_df.to_csv(summary_path, index=False)
    aa_df.to_csv(distribution_path)
    
    end_time = time.time()
    print(f"--- UniRef50 statistics calculation finished in {end_time - start_time:.2f} seconds ---")
    print(f"Summary stats saved to {summary_path}")
    print(f"Amino acid distribution saved to {distribution_path}")
    results_volume.commit()


@app.function(
    image=image,
    mounts=mounts,
    volumes={"/models": model_volume, "/results": results_volume},
    timeout=3600,
)
def run_sanity_checks_and_stats():
    """Run a suite of sanity checks and calculate dataset statistics."""
    
    # --- Start statistics calculation in the background ---
    stats_future = calculate_uniref50_stats.remote()

    # Add project directories to Python path for correct module resolution
    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root")

    from sanity_check.run_checks import (
        get_shuffled_vs_real_protein,
        get_low_complexity_vs_real_protein,
        get_helix_vs_real_protein,
        get_sheet_vs_real_protein,
        get_disordered_vs_real_protein,
        get_positional_motifs_protein,
        run_benchmark
    )
    from gensim.models import FastText

    FIGURES_DIR = Path("/results/sanity_checks_protein/figures")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    models_to_check = {
        "k-mer_model": {
            "path": "/models/uniref50_pretrained_10pct_kmer.epoch6.bin",
            "k": 5,
            "tokenization": "kmer",
        },
        "residue_model": {
            "path": "/models/uniref50_pretrained_full_residue.bin",
            "k": 1, 
            "tokenization": "residue",
        }
    }
    
    benchmarks = [
        (get_shuffled_vs_real_protein, "Shuffled vs. Real Proteins"),
        (get_low_complexity_vs_real_protein, "Low-Complexity vs. Real Proteins"),
        (get_helix_vs_real_protein, "Helix-like vs. Real Proteins"),
        (get_sheet_vs_real_protein, "Sheet-like vs. Real Proteins"),
        (get_disordered_vs_real_protein, "Disordered-like vs. Real Proteins"),
        (get_positional_motifs_protein, "Positional Motif Detection"),
    ]

    for model_name, model_info in models_to_check.items():
        print(f"\n--- Running checks for {model_name} ---")
        model_path = Path(model_info['path'])
        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}. Skipping.")
            continue
        
        model = FastText.load(str(model_path))
        
        for fn, name in benchmarks:
            run_benchmark(model, fn, name, model_info['k'], FIGURES_DIR, tokenization=model_info['tokenization'])

    # Generate PDF report
    pdf = PDF()
    report_path = "/results/sanity_checks_protein/protein_sanity_checks_report.pdf"
    
    all_plots = list(FIGURES_DIR.glob("*.png"))
    interpretations = {
        "shuffled_vs._real_proteins": "Tests if the model can distinguish real proteins from shuffled versions. A good model should form distinct clusters, indicating it has learned structural or sequential information beyond mere amino acid composition.",
        "low-complexity_vs._real_proteins": "Tests if the model can distinguish real, complex proteins from simple, repetitive sequences. Distinct clusters suggest the model has learned about protein complexity.",
        "helix-like_vs._real_proteins": "Compares real proteins to synthetic sequences composed of helix-forming amino acids. Separation indicates the model has learned sequence patterns associated with alpha-helices.",
        "sheet-like_vs._real_proteins": "Compares real proteins to synthetic sequences composed of beta-sheet-forming amino acids. Separation indicates the model has learned sequence patterns associated with beta-sheets.",
        "disordered-like_vs._real_proteins": "Compares real proteins to synthetic sequences composed of disorder-promoting amino acids. Separation suggests the model can identify patterns related to intrinsically disordered regions.",
        "positional_motif_detection": "Tests if positional encoders can differentiate the same motif based on its location (start, middle, end). Models with effective PEs should show clear separation between these groups.",
    }

    for path in sorted(all_plots):
        title = path.stem.replace('_', ' ').title()
        base_name = "_".join(path.stem.split('_')[:-2]) # Get base benchmark name
        interpretation = interpretations.get(base_name, "No interpretation available.")
        pdf.add_plot(str(path), title, interpretation)
            
    pdf.output(report_path, 'F')
    results_volume.commit()
    print(f"\nSanity check report saved to {report_path}")
    
    # Wait for the stats calculation to finish if it hasn't already
    modal.container_app.functions.calculate_uniref50_stats.wait(stats_future)
    print("\n--- All tasks completed successfully! ---")

if __name__ == "__main__":
    # This script is intended to be run via `modal run`
    pass 