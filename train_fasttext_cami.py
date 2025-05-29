import os
import gzip
import fasttext
from Bio import SeqIO # Using Bio.SeqIO for easy FASTA reading

def kmerize(seq, k):
    """Yield k-mers from a sequence."""
    for i in range(len(seq) - k + 1):
        yield seq[i:i+k]

def cami_fasta_kmer_iter(file_paths, k=5, chunk_size=100000):
    """Generator to read k-mers from a list of gzipped FASTA files."""
    print(f"Reading sequences from {file_paths} and generating k-mers...")
    kmer_buffer = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}. Skipping.")
            continue
        try:
            with gzip.open(file_path, "rt") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    seq = str(record.seq).upper()
                    for kmer in kmerize(seq, k):
                        kmer_buffer.append(kmer)
                        if len(kmer_buffer) >= chunk_size:
                            yield " ".join(kmer_buffer)
                            kmer_buffer = []
        except Exception as e:
            print(f"Error reading or processing file {file_path}: {e}")
            # Decide if you want to stop or continue with other files
            # raise e # Uncomment to stop on error
    # Yield any remaining k-mers in the buffer
    if kmer_buffer:
        yield " ".join(kmer_buffer)
    print("Finished reading all files.")

# Define the paths to the downloaded CAMI II FASTA files
# Assuming they were downloaded into CAMI_Airways and CAMI_Skin directories
cami_files = [
    "CAMI_Airways/anonymous_gsa_pooled.fasta.gz",
    "CAMI_Skin/anonymous_gsa_pooled.fasta.gz",
]

# Training parameters
k = 5
dim = 100 # Reduced dimension for a toy model
workers = 8 # Adjust based on your system
epochs = 5 # Reduced epochs for a toy model
output = "fasttext_cami_toy.bin"

print("Starting FastText training on CAMI II toy dataset (DNA)...")
print(f"Parameters: k={k}, dim={dim}, workers={workers}, epochs={epochs}")

# FastText requires training data in a file or a list of sentences.
# For a generator, we can write to a temporary file or process in chunks.
# Given the toy size, we can create a list of sentences (chunks).

# Collect k-mer chunks from the generator
# Adjust chunk_size based on available memory. 100000 k-mers is a reasonable chunk.
# For a very small dataset, you might even collect all into one list.

# We'll train using the skipgram model on the generated k-mers
# FastText unsupervised training takes a file path or uses the `words` argument.
# The `words` argument expects a list of strings where each string is a 'sentence'.
# Here, a 'sentence' is a collection of k-mers from a chunk of sequences.

try:
    # Collect k-mer chunks into a list of 'sentences' for training
    # For a real large dataset, you would stream directly from files if the library supports it
    # or write to a temporary file and pass the file path to fasttext.

    # Let's collect all k-mers for this small toy example
    # For larger datasets, use a temporary file approach
    all_kmers = []
    for chunk_of_kmers_str in cami_fasta_kmer_iter(cami_files, k=k, chunk_size=50000): # Smaller chunk size for collecting
         all_kmers.extend(chunk_of_kmers_str.split())

    # FastText train_unsupervised can take a list of words or sentences directly
    # We'll pass the list of k-mers (words)
    print(f"Collected {len(all_kmers)} k-mers for training.")

    # FastText expects training data as a list of lists of strings or a file path
    # Passing a list of all k-mers as a single 'sentence' for simplicity in toy example
    # For real world, consider sentence boundaries (e.g., per sequence or chunks)
    sentences = [all_kmers]

    print("Starting FastText unsupervised training...")
    # Using skipgram model for unsupervised training
    model = fasttext.train_unsupervised(
        model='skipgram',
        dim=dim,
        epoch=epochs,
        thread=workers,
        minCount=1, # Keep all k-mers in toy model
        verbose=2,
        # Pass sentences directly
        # 'words' argument is not standard in train_unsupervised, should use file path
        # Let's write to a temporary file for training.
        # This is more representative of handling larger datasets.
    )

    # Write collected k-mers to a temporary file for FastText training
    temp_train_file = "cami_train_data.txt"
    with open(temp_train_file, "w") as f:
        # Write each chunk as a line/sentence, or just all k-mers separated by spaces
        # For this toy, writing all k-mers separated by spaces is simple
        f.write(" ".join(all_kmers))

    print(f"Training FastText model on {temp_train_file}...")
    model = fasttext.train_unsupervised(
        temp_train_file,
        model='skipgram',
        dim=dim,
        epoch=epochs,
        thread=workers,
        minCount=1,
        verbose=2
    )


    print(f"Training complete. Saving model to {output}")
    model.save(output)
    if os.path.exists(output):
        print(f"Successfully saved model to {output}")
        print(f"Model file size: {os.path.getsize(output) / (1024*1024):.2f} MB")
    else:
        print("Warning: Model file was not created!")

    # Clean up temporary file
    # os.remove(temp_train_file)
    # print(f"Removed temporary training file: {temp_train_file}")

except Exception as e:
    print(f"Error during training: {str(e)}")
    # raise e # Uncomment to re-raise the exception 