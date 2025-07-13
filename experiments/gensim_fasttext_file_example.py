import logging
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import datapath

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss_previous_step = 0.0
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print(f"[word2vec] epoch {self.epoch}: loss={loss - self.loss_previous_step:.2f}")
        self.loss_previous_step = loss
        self.epoch += 1

def main():
    # Use the Lee Corpus as in the Gensim documentation
    corpus_file = datapath('lee_background.cor')
    print(f"Using corpus file: {corpus_file}")

    # --- FastText Example (no loss tracking) ---
    print("\n--- FastText Example (no loss tracking) ---")
    ft_model = FastText(vector_size=100)
    ft_model.build_vocab(corpus_file=corpus_file)
    print(f"FastText vocab size: {len(ft_model.wv)}")
    ft_model.train(
        corpus_file=corpus_file,
        epochs=ft_model.epochs,
        total_examples=ft_model.corpus_count,
        total_words=ft_model.corpus_total_words,
    )
    print(ft_model)
    try:
        loss = ft_model.get_latest_training_loss()
        print(f"FastText latest training loss: {loss}")
    except Exception as e:
        print(f"FastText loss tracking not available: {e}")

    # --- Word2Vec Example (with loss tracking) ---
    print("\n--- Word2Vec Example (with loss tracking) ---")
    w2v_model = Word2Vec(vector_size=100, compute_loss=True)
    w2v_model.build_vocab(corpus_file=corpus_file)
    print(f"Word2Vec vocab size: {len(w2v_model.wv)}")
    w2v_model.train(
        corpus_file=corpus_file,
        epochs=w2v_model.epochs,
        total_examples=w2v_model.corpus_count,
        total_words=w2v_model.corpus_total_words,
        compute_loss=True,
        callbacks=[LossLogger()],
    )
    print(w2v_model)
    try:
        loss = w2v_model.get_latest_training_loss()
        print(f"Word2Vec latest training loss: {loss}")
    except Exception as e:
        print(f"Word2Vec loss tracking not available: {e}")

if __name__ == "__main__":
    main() 