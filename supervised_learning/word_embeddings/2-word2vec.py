#!/usr/bin/env python3
""" gensim """
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5, 
                  negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """ Create, build and train a gensim Word2Vec model """

    # set sg parameter (0 for CBOW, 1 for Skip-gram)
    sg = 0 if cbow else 1

    # create Word2Vec model
    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative,
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
