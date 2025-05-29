import os
from datasets import load_dataset
from pe_fasttext.fasttext_utils import train_fasttext


def kmerize(seq, k):
    """Yield k-mers from a sequence."""
    for i in range(len(seq) - k + 1):
        yield seq[i:i+k]

def og_dna_corpus_iter(k=5, chunk_size=1000000, max_kmers=50000):
    print("Streaming OG dataset from Hugging Face (DNA mode)...")
    ds = load_dataset('tattabio/OG', streaming=True)['train']
    chunk = []
    total_kmers = 0
    for row in ds:
        for dna_seq in row['IGS_seqs']:
            for kmer in kmerize(dna_seq, k):
                chunk.append(kmer)
                total_kmers += 1
                if total_kmers >= max_kmers:
                    print(f"Collected {len(chunk)} k-mers (reached max_kmers={max_kmers})")
                    yield chunk
                    return
    if chunk:
        print(f"Processing final chunk of {len(chunk)} k-mers (total processed: {total_kmers})")
        yield chunk

# Training parameters
k = 5
dim = 512
workers = 32
epochs = 10
output = "fasttext_dna_og.bin"

print("Starting FastText training on OG dataset (DNA)... (first 50,000 k-mers)")
print(f"Parameters: k={k}, dim={dim}, workers={workers}, epochs={epochs}")

try:
    # Collect all k-mers into a list of sentences (each chunk is a sentence)
    kmer_chunks = list(og_dna_corpus_iter(k, max_kmers=50000))
    model = train_fasttext(
        kmer_chunks,  # each chunk is a sentence
        vector_size=dim,
        workers=workers,
        epochs=epochs
    )
    print(f"Training complete. Saving model to {output}")
    model.save(output)
    if os.path.exists(output):
        print(f"Successfully saved model to {output}")
        print(f"Model file size: {os.path.getsize(output) / (1024*1024):.2f} MB")
    else:
        print("Warning: Model file was not created!")
except Exception as e:
    print(f"Error during training: {str(e)}")
    raise 