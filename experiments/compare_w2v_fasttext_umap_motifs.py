import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, FastText
from sklearn.preprocessing import StandardScaler
import umap
from datasets import load_dataset
from adjustText import adjust_text
from pe_fasttext.position_encodings import ENCODERS

def kmerize(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def get_og_kmer_sentences(k=5, max_sequences=100):
    ds = load_dataset('tattabio/OG', streaming=True)['train']
    sentences = []
    for row in ds:
        for seq in row['IGS_seqs']:
            kmers = kmerize(seq, k)
            if kmers:
                sentences.append(kmers)
                if len(sentences) >= max_sequences:
                    return sentences
    return sentences

def generate_motif_family(label, motif, flank_base, n_seqs=10, seq_len=100):
    seqs = []
    motif_len = len(motif)
    flank_len = (seq_len - motif_len) // 2
    for _ in range(n_seqs):
        left = "".join(np.random.choice(list(flank_base), flank_len))
        right = "".join(np.random.choice(list(flank_base), seq_len - motif_len - flank_len))
        seq = left + motif + right
        seqs.append((label, seq))
    return seqs

def get_original_motif_families():
    families = [
        ("polyA", "A" * 20, "CGT"),
        ("polyT", "T" * 20, "ACG"),
        ("polyC", "C" * 20, "ATG"),
        ("polyG", "G" * 20, "ACT"),
        ("AT_repeat", "AT" * 10, "CG"),
        ("GC_repeat", "GC" * 10, "AT"),
        ("motif1", "CGTACGTACG", "ATGC"),
        ("motif2", "TATAATATAA", "GCGC"),
        ("motif3", "GCGCGCGCGC", "ATAT"),
    ]
    all_seqs = []
    for label, motif, flank_base in families:
        all_seqs.extend(generate_motif_family(label, motif, flank_base, n_seqs=10, seq_len=100))
    return all_seqs

def get_refined_position_motifs():
    seq_len = 100
    motif = "GATTACA"
    tata = "TATAAA"
    caat = "CAAT"
    palindrome = "AGCTTTCGA"
    families = [
        (f"{motif}_start", lambda: motif + "".join(np.random.choice(list("ACGT"), seq_len - len(motif)))),
        (f"{motif}_middle", lambda: "".join(np.random.choice(list("ACGT"), (seq_len - len(motif)) // 2)) + motif + "".join(np.random.choice(list("ACGT"), seq_len - len(motif) - (seq_len - len(motif)) // 2))),
        (f"{motif}_end", lambda: "".join(np.random.choice(list("ACGT"), seq_len - len(motif))) + motif),
        (f"{tata}_10", lambda: "".join(np.random.choice(list("ACGT"), 10)) + tata + "".join(np.random.choice(list("ACGT"), seq_len - 10 - len(tata)))),
        (f"{tata}_50", lambda: "".join(np.random.choice(list("ACGT"), 50)) + tata + "".join(np.random.choice(list("ACGT"), seq_len - 50 - len(tata)))),
        (f"{tata}_90", lambda: "".join(np.random.choice(list("ACGT"), 90)) + tata + "".join(np.random.choice(list("ACGT"), seq_len - 90 - len(tata)))),
        ("palindrome_center", lambda: "".join(np.random.choice(list("ACGT"), 45)) + palindrome + palindrome[::-1] + "".join(np.random.choice(list("ACGT"), 45))),
        ("palindrome_early", lambda: "".join(np.random.choice(list("ACGT"), 10)) + palindrome + palindrome[::-1] + "".join(np.random.choice(list("ACGT"), 80))),
        ("GCGCGC_flank_A", lambda: "A"*10 + "GCGCGC" + "".join(np.random.choice(list("ACGT"), seq_len - 16))),
        ("GCGCGC_flank_T", lambda: "T"*10 + "GCGCGC" + "".join(np.random.choice(list("ACGT"), seq_len - 16))),
        (f"{tata}_10_90", lambda: "".join(np.random.choice(list("ACGT"), 10)) + tata + "".join(np.random.choice(list("ACGT"), 74)) + tata + "".join(np.random.choice(list("ACGT"), 4))),
        (f"{tata}_30_70", lambda: "".join(np.random.choice(list("ACGT"), 30)) + tata + "".join(np.random.choice(list("ACGT"), 34)) + tata + "".join(np.random.choice(list("ACGT"), 24))),
        (f"{caat}_every_10", lambda: "".join([caat + "".join(np.random.choice(list("ACGT"), 6)) for _ in range(10)])),
        (f"{caat}_every_20", lambda: "".join([caat + "".join(np.random.choice(list("ACGT"), 16)) for _ in range(5)])),
    ]
    all_seqs = []
    n_seqs = 10
    for label, seq_fn in families:
        for _ in range(n_seqs):
            all_seqs.append((label, seq_fn()))
    return all_seqs

def get_random_sequences(n=100, min_len=80, max_len=120):
    seqs = []
    for i in range(n):
        seq_len = np.random.randint(min_len, max_len + 1)
        seq = "".join(np.random.choice(list("ACGT"), seq_len))
        seqs.append(("random", seq))
    return seqs

def embed_sequences(model, sentences):
    vectors = []
    for sent in sentences:
        vecs = [model.wv[kmer] for kmer in sent if kmer in model.wv]
        if vecs:
            vectors.append(np.mean(vecs, axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))
    return np.vstack(vectors)

def embed_sequences_pefasttext(model, sentences, pos_encoder, fusion="add"):
    vectors = []
    for sent in sentences:
        valid_indices = [i for i, kmer in enumerate(sent) if kmer in model.wv]
        if not valid_indices:
            if fusion == "add":
                vectors.append(np.zeros(model.vector_size))
            elif fusion == "concat":
                vectors.append(np.zeros(model.vector_size * 2))
            continue
        embs = np.array([model.wv[sent[i]] for i in valid_indices])
        poses = pos_encoder(valid_indices)
        if fusion == "add":
            vecs = embs + poses
        elif fusion == "concat":
            vecs = np.concatenate([embs, poses], axis=1)
        else:
            raise ValueError("fusion must be 'add' or 'concat'")
        vectors.append(np.mean(vecs, axis=0))
    return np.vstack(vectors)

def plot_umaps(embeds_dict, labels, unique_labels, out_path, title_prefix):
    n_models = len(embeds_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6))
    if n_models == 1:
        axes = [axes]
    color_map = {lab: i for i, lab in enumerate(unique_labels)}
    cmap = plt.get_cmap('tab10')
    label_to_color = {lab: cmap(color_map[lab] / max(1, len(unique_labels)-1)) for lab in unique_labels}
    for ax, (model_name, umap_embeds) in zip(axes, embeds_dict.items()):
        scatter = ax.scatter(umap_embeds[:, 0], umap_embeds[:, 1], c=[label_to_color[lab] for lab in labels], s=60, alpha=0.8)
        ax.set_title(f"{title_prefix} {model_name}")
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=lab, markerfacecolor=label_to_color[lab], markersize=8)
               for lab in unique_labels]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=600)
    plt.close(fig)

def get_tata_variable_spacing_benchmark(seq_len=100, n_seqs=10):
    motif = "TATAAA"
    spacings = [5, 10, 20, 40, 60, 80]
    families = []
    for spacing in spacings:
        label = f"TATAAA_spaced_{spacing}"
        def make_seq(spacing=spacing):
            before = np.random.choice(list("ACGT"), seq_len - 2*len(motif) - spacing)
            return motif + "".join(np.random.choice(list("ACGT"), spacing)) + motif + "".join(before)
        families.append((label, make_seq))
    all_seqs = []
    for label, seq_fn in families:
        for _ in range(n_seqs):
            all_seqs.append((label, seq_fn()))
    return all_seqs

def run_benchmark_tata_variable_spacing(models, k, vector_size, out_path):
    motif_seqs = get_tata_variable_spacing_benchmark()
    motif_labels = [label for label, seq in motif_seqs]
    motif_sequences = [seq for label, seq in motif_seqs]
    motif_sentences = [kmerize(seq, k) for seq in motif_sequences]
    unique_labels = sorted(set(motif_labels))
    embeds_dict = {}
    # Standard models
    for name, model in models.items():
        if name.startswith("PE-"):
            continue
        embeds = embed_sequences(model, motif_sentences)
        embeds_std = StandardScaler().fit_transform(embeds)
        umap_embeds = umap.UMAP(random_state=42).fit_transform(embeds_std)
        embeds_dict[name] = umap_embeds
    # PE models
    for name, value in models.items():
        if not name.startswith("PE-"):
            continue
        model, pos_encoder = value
        embeds = embed_sequences_pefasttext(model, motif_sentences, pos_encoder, fusion="add")
        embeds_std = StandardScaler().fit_transform(embeds)
        umap_embeds = umap.UMAP(random_state=42).fit_transform(embeds_std)
        embeds_dict[name] = umap_embeds
    plot_umaps(embeds_dict, motif_labels, unique_labels, out_path, "TATA Motif Variable Spacing")

def get_mixed_spacing_content_benchmark(seq_len=100, n_seqs=10):
    families = [
        ("TATAAA_10_CAAT_50", lambda: "".join(np.random.choice(list("ACGT"), 10)) + "TATAAA" + "".join(np.random.choice(list("ACGT"), 34)) + "CAAT" + "".join(np.random.choice(list("ACGT"), seq_len - 10 - 6 - 34 - 4))),
        ("CAAT_10_TATAAA_50", lambda: "".join(np.random.choice(list("ACGT"), 10)) + "CAAT" + "".join(np.random.choice(list("ACGT"), 34)) + "TATAAA" + "".join(np.random.choice(list("ACGT"), seq_len - 10 - 4 - 34 - 6))),
        ("TATAAA_10_CAAT_90", lambda: "".join(np.random.choice(list("ACGT"), 10)) + "TATAAA" + "".join(np.random.choice(list("ACGT"), 74)) + "CAAT" + "".join(np.random.choice(list("ACGT"), seq_len - 10 - 6 - 74 - 4))),
        ("TATAAA_10_TATAAA_50", lambda: "".join(np.random.choice(list("ACGT"), 10)) + "TATAAA" + "".join(np.random.choice(list("ACGT"), 34)) + "TATAAA" + "".join(np.random.choice(list("ACGT"), seq_len - 10 - 6 - 34 - 6))),
        ("CAAT_10_CAAT_50", lambda: "".join(np.random.choice(list("ACGT"), 10)) + "CAAT" + "".join(np.random.choice(list("ACGT"), 34)) + "CAAT" + "".join(np.random.choice(list("ACGT"), seq_len - 10 - 4 - 34 - 4))),
    ]
    all_seqs = []
    for label, seq_fn in families:
        for _ in range(n_seqs):
            all_seqs.append((label, seq_fn()))
    return all_seqs

def run_benchmark_mixed_spacing_content(models, k, vector_size, out_path):
    motif_seqs = get_mixed_spacing_content_benchmark()
    motif_labels = [label for label, seq in motif_seqs]
    motif_sequences = [seq for label, seq in motif_seqs]
    motif_sentences = [kmerize(seq, k) for seq in motif_sequences]
    unique_labels = sorted(set(motif_labels))
    embeds_dict = {}
    # Standard models
    for name, model in models.items():
        if name.startswith("PE-"):
            continue
        embeds = embed_sequences(model, motif_sentences)
        embeds_std = StandardScaler().fit_transform(embeds)
        umap_embeds = umap.UMAP(random_state=42).fit_transform(embeds_std)
        embeds_dict[name] = umap_embeds
    # PE models
    for name, value in models.items():
        if not name.startswith("PE-"):
            continue
        model, pos_encoder = value
        embeds = embed_sequences_pefasttext(model, motif_sentences, pos_encoder, fusion="add")
        embeds_std = StandardScaler().fit_transform(embeds)
        umap_embeds = umap.UMAP(random_state=42).fit_transform(embeds_std)
        embeds_dict[name] = umap_embeds
    plot_umaps(embeds_dict, motif_labels, unique_labels, out_path, "Motif Mixed Spacing/Content")

def run_benchmark_real_vs_random(models, k, vector_size, og_sentences, out_path, n_real=100, min_len=80, max_len=120):
    n_real = min(n_real, len(og_sentences))
    real_labels = ["real"] * n_real
    real_sequences = ["".join(seq) for seq in og_sentences[:n_real]]
    random_seqs = get_random_sequences(n=n_real, min_len=min_len, max_len=max_len)
    random_labels = [label for label, seq in random_seqs]
    random_sequences = [seq for label, seq in random_seqs]
    all_labels = real_labels + random_labels
    all_sequences = real_sequences + random_sequences
    all_sentences = [kmerize(seq, k) for seq in all_sequences]
    unique_labels = sorted(set(all_labels))
    embeds_dict = {}
    # Standard models
    for name, model in models.items():
        if name.startswith("PE-"):
            continue
        embeds = embed_sequences(model, all_sentences)
        embeds_std = StandardScaler().fit_transform(embeds)
        umap_embeds = umap.UMAP(random_state=42).fit_transform(embeds_std)
        embeds_dict[name] = umap_embeds
    # PE models
    for name, value in models.items():
        if not name.startswith("PE-"):
            continue
        model, pos_encoder = value
        embeds = embed_sequences_pefasttext(model, all_sentences, pos_encoder, fusion="add")
        embeds_std = StandardScaler().fit_transform(embeds)
        umap_embeds = umap.UMAP(random_state=42).fit_transform(embeds_std)
        embeds_dict[name] = umap_embeds
    plot_umaps(embeds_dict, all_labels, unique_labels, out_path, "Real vs Random")

def run_benchmark_original_motifs(models, k, vector_size, out_path):
    motif_seqs = get_original_motif_families()
    motif_labels = [label for label, seq in motif_seqs]
    motif_sequences = [seq for label, seq in motif_seqs]
    motif_sentences = [kmerize(seq, k) for seq in motif_sequences]
    unique_labels = sorted(set(motif_labels))
    embeds_dict = {}
    # Standard models
    for name, model in models.items():
        if name.startswith("PE-"):
            continue
        embeds = embed_sequences(model, motif_sentences)
        embeds_std = StandardScaler().fit_transform(embeds)
        umap_embeds = umap.UMAP(random_state=42).fit_transform(embeds_std)
        embeds_dict[name] = umap_embeds
    # PE models
    for name, value in models.items():
        if not name.startswith("PE-"):
            continue
        model, pos_encoder = value
        embeds = embed_sequences_pefasttext(model, motif_sentences, pos_encoder, fusion="add")
        embeds_std = StandardScaler().fit_transform(embeds)
        umap_embeds = umap.UMAP(random_state=42).fit_transform(embeds_std)
        embeds_dict[name] = umap_embeds
    plot_umaps(embeds_dict, motif_labels, unique_labels, out_path, "Motif")

def main():
    k = 10
    vector_size = 128
    window = 5
    min_count = 1
    epochs = 10

    print("Loading and tokenizing OG dataset...")
    all_og_sentences = get_og_kmer_sentences(k=k, max_sequences=1100)
    print(f"Loaded {len(all_og_sentences)} OG sequences.")

    # Split: first 1,000 for training, next 100 for real benchmark
    train_og_sentences = all_og_sentences[:1000]
    heldout_og_sentences = all_og_sentences[1000:1100]

    # Compute 5th and 95th percentiles for sequence lengths from training data
    train_lengths = [len(seq) + k - 1 for seq in train_og_sentences if len(seq) > 0]
    min_len = int(np.percentile(train_lengths, 5))
    max_len = int(np.percentile(train_lengths, 95))
    print(f"Trimmed training sequence length range: {min_len} to {max_len}")

    print("Training Word2Vec...")
    w2v = Word2Vec(train_og_sentences, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, workers=1)
    print("Training FastText...")
    ft = FastText(train_og_sentences, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, workers=1)

    # Prepare all positional encodings
    pe_models = {}
    for enc_name, enc_cls in ENCODERS.items():
        pos_encoder = enc_cls(vector_size)
        pe_models[f"PE-{enc_name}"] = (ft, pos_encoder)

    # Collect all models
    models = {
        "Word2Vec": w2v,
        "FastText": ft,
        **pe_models
    }

    # Benchmark 1: Original motif clustering (up to motif3)
    run_benchmark_original_motifs(models, k, vector_size, "plots/original_motif_clustering_umap.png")
    print("Saved original motif clustering UMAP to plots/original_motif_clustering_umap.png")

    # Benchmark 2: Motif mixed spacing/content
    run_benchmark_mixed_spacing_content(models, k, vector_size, "plots/motif_mixed_spacing_content_umap.png")
    print("Saved mixed spacing/content motif UMAP to plots/motif_mixed_spacing_content_umap.png")

    # Benchmark 3: Real vs random (use held-out OG sequences, match random seq length to trimmed training range)
    run_benchmark_real_vs_random(models, k, vector_size, heldout_og_sentences, "plots/real_vs_random_umap.png", n_real=100, min_len=min_len, max_len=max_len)
    print("Saved real vs random UMAP to plots/real_vs_random_umap.png")

if __name__ == "__main__":
    main() 