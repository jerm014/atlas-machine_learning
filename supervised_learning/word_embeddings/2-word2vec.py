#!/usr/bin/env python3
""" gensim """
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5, 
                  negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """ Create, build and train a gensim Word2Vec model """
    # preprocess to get words
    # preprocessed_sentences = []
    # for sentence in sentences:
    #    clean_sentence = re.sub(r'\'s\b|\'\b', '', sentence.lower())
    #    clean_sentence = re.sub(r'[^\w\s]', '', clean_sentence)
    #    words = clean_sentence.split()
    #    preprocessed_sentences.append(words)

    # set sg parameter (0 for CBOW, 1 for Skip-gram)
    sg = 0 if cbow else 1

    # create and train the Word2Vec model
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
