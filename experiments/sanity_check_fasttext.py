import os
from gensim.models.fasttext import FastText
from pe_fasttext.fasttext_utils import LossLogger

def main():
    # Toy dataset: 100 sentences, each with 8 tokens
    toy_data = [["A", "C", "G", "T", "A", "C", "G", "T"] for _ in range(100)]
    print(f"Toy data: {len(toy_data)} sentences, first: {toy_data[0]}")

    model = FastText(
        vector_size=50,
        window=2,
        min_count=1,
        sg=1,
        workers=1,
    )
    model.build_vocab(toy_data)
    print(f"Vocab size: {len(model.wv)}")
    model.train(
        toy_data,
        total_examples=model.corpus_count,
        epochs=5,
        compute_loss=True,
        callbacks=[LossLogger()],
    )
    print("Sanity check complete.")

if __name__ == "__main__":
    main() 