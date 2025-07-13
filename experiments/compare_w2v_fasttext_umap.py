import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from gensim.models import Word2Vec, FastText
from sklearn.preprocessing import StandardScaler
import umap

from pe_fasttext.utils import kmerize

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

def embed_sequences(model, sentences):
    """Embed pre-tokenized sentences (list of k-mers)."""
    vectors = []
    for sent in sentences:
        # Only use k-mers in vocab
        vecs = [model.wv[kmer] for kmer in sent if kmer in model.wv]
        if vecs:
            vectors.append(np.mean(vecs, axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))
    return np.vstack(vectors)

def main():
    k = 5
    max_sequences = 100
    vector_size = 64
    window = 5
    min_count = 1
    epochs = 10

    print("Loading and tokenizing OG dataset...")
    sentences = get_og_kmer_sentences(k=k, max_sequences=max_sequences)
    print(f"Loaded {len(sentences)} sequences.")

    print("Training Word2Vec...")
    w2v = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, workers=1)
    print("Training FastText...")
    ft = FastText(sentences, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, workers=1)

    print("Embedding sequences...")
    w2v_embeds = embed_sequences(w2v, sentences)
    ft_embeds = embed_sequences(ft, sentences)

    # Standardize before UMAP
    scaler = StandardScaler()
    w2v_embeds_std = scaler.fit_transform(w2v_embeds)
    ft_embeds_std = scaler.fit_transform(ft_embeds)

    print("Running UMAP...")
    reducer = umap.UMAP(random_state=42)
    w2v_umap = reducer.fit_transform(w2v_embeds_std)
    ft_umap = reducer.fit_transform(ft_embeds_std)

    print("Plotting...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(w2v_umap[:, 0], w2v_umap[:, 1], c='blue', alpha=0.7)
    axes[0].set_title("Word2Vec UMAP")
    axes[1].scatter(ft_umap[:, 0], ft_umap[:, 1], c='green', alpha=0.7)
    axes[1].set_title("FastText UMAP")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 