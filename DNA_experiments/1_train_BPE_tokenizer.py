from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from collections import Counter, defaultdict, deque


FASTA_PATH = "./data/GCA_000001405.15_GRCh38_genomic.fna"
BPE_MERGES = 4096 # same vocabulary size as k-mer model
# 200k chunks * 5k chunk size = 1M BP coverage -> enough for tokenizer.
# The full FastText model will have more coverage
CHUNK_SIZE = 5000
MAX_CHUNKS = 200000 

def sequence_stream_iterator(fasta_path, chunk_size=CHUNK_SIZE, max_chunks=None):
    """Stream raw DNA sequences from FASTA file for tokenizer training."""
    count = 0
    with open(fasta_path, "r") as fh:
        seq = []
        for line in fh:
            line = line.upper()

            if line.startswith(">"):
                if seq:
                    seq_str = "".join(seq)
                    for i in range(0, len(seq_str), chunk_size):
                        chunk = seq_str[i:i+chunk_size]
                        if len(chunk) > 10:
                            yield chunk
                            count += 1
                            if count % 10000 == 0:
                                print(f"Streamed {count} raw chunks for tokenizer training...")
                            if max_chunks is not None and count >= max_chunks:
                                print(f"Reached max_chunks={max_chunks} for tokenizer training.")
                                return
                    seq = []
            else:
                seq.append(line.strip())
        
        if seq:
            seq_str = "".join(seq)
            for i in range(0, len(seq_str), chunk_size):
                chunk = seq_str[i:i+chunk_size]
                if len(chunk) > 10:
                    yield chunk
                    count += 1
                    if count % 10000 == 0:
                        print(f"Streamed {count} raw chunks for tokenizer training...")
                    if max_chunks is not None and count >= max_chunks:
                        print(f"Reached max_chunks={max_chunks} for tokenizer training.")
                        return

    print(f"Finished streaming {count} raw chunks for tokenizer training.")

def main():
    # Define the BPE tokenizer
    bpe = Tokenizer(models.BPE())
    # pre-tokenizer
    bpe.pre_tokenizer = pre_tokenizers.Whitespace()
    # trainer
    trainer = trainers.BpeTrainer(
        vocab_size=BPE_MERGES, 
        min_frequency=10, 
        limit_alphabet=4, 
        initial_alphabet=["A", "C", "G", "T"], 
        special_tokens=["[UNK]", "</w>"], 
        show_progress=True,
    )

    # iterate over the corpus and train
    print("Starting BPE tokenizer training...")
    corpus_iter = lambda: sequence_stream_iterator(FASTA_PATH, chunk_size=CHUNK_SIZE, max_chunks=MAX_CHUNKS)
    bpe.train_from_iterator(corpus_iter(), trainer=trainer)

    # save the tokenizer 
    bpe.save("hg38_tokenizer.json")
    print("BPE tokenizer training complete.")

if __name__ == "__main__":
    main() 