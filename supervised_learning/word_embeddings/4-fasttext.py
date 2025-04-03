#!/usr/bin/env python3
""" fasttext """
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """ Create, build and train a gensim Word2Vec model """

    # set sg parameter (0 for CBOW, 1 for Skip-gram)
    sg = 0 if cbow else 1

    # create fasttext model
    model = gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        workers=workers,
        negative=negative,
        sg=sg,
        seed=seed
    )

    # build vocab
    model.build_vocab(sentences)

    # train the model
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
