import os
from pe_fasttext.tokenization import fasta_stream
from pe_fasttext.fasttext_utils import train_fasttext

def corpus_iter(corpus_path, k):
    print(f"Reading sequences from {corpus_path}...")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Could not find corpus file: {corpus_path}")
    
    # Instead of loading all tokens at once, yield them in chunks
    chunk = []
    chunk_size = 1000000  # Process 1M k-mers at a time
    total_kmers = 0
    
    for token in fasta_stream(corpus_path, k):
        chunk.append(token)
        if len(chunk) >= chunk_size:
            total_kmers += len(chunk)
            print(f"Processing chunk of {len(chunk)} k-mers (total processed: {total_kmers})")
            yield chunk
            chunk = []
    
    # Don't forget the last chunk
    if chunk:
        total_kmers += len(chunk)
        print(f"Processing final chunk of {len(chunk)} k-mers (total processed: {total_kmers})")
        yield chunk

# Training parameters
corpus_path = "data/uniref50.fasta"
k = 5
dim = 512
workers = 32
epochs = 10
output = "fasttext_protein.bin"

print("Starting FastText training...")
print(f"Parameters: k={k}, dim={dim}, workers={workers}, epochs={epochs}")

try:
    model = train_fasttext(
        corpus_iter(corpus_path, k),
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