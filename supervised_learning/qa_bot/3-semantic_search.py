#!/usr/bin/env python3
"""
semantic search over a corpus of documents for task 3.

this module exposes semantic_search(corpus_path, sentence) – it returns
the text of the document (in corpus_path) most semantically similar to the
input sentence using embeddings from the sentence encoder.
"""
import os
import sys
import numpy as np
import tensorflow_hub as hub

# use a global cache so the model loads only once...
ENCODER = None


def semantic_search(corpus_path, sentence):
    """return the text of the document most similar to the sentence.

    Args:
        corpus_path: directory containing plain‑text documents.
        sentence: query string.

    Returns:
        str or None
    """
    # get the list of files
    files = list_files(corpus_path)
    if not files:
        return None

    # read all those files
    text_blobs = read_files(files)
    if not text_blobs:
        return None

    model = load_encoder()

    embeddings = model([sentence] + text_blobs).numpy()
    query_vec, doc_vecs = embeddings[0], embeddings[1:]

    norm_query = np.linalg.norm(query_vec)
    norm_docs = np.linalg.norm(doc_vecs, axis=1)
    s1 = np.dot(doc_vecs, query_vec)
    s2 = norm_docs * norm_query + 1e-9
    similarities = s1 / s2

    # max similarities is index of document most similar to the sentence
    best_idx = int(np.argmax(similarities))
    return text_blobs[best_idx]


def list_files(directory):
    """return a list of file paths in a directory (not recursive)"""
    try:
        entries = os.listdir(directory)
    except OSError as exc:
        sys.exit(f"error: cannot access this path: {exc}")
    return [os.path.join(directory, name) for name in entries
            if os.path.isfile(os.path.join(directory, name))]


def read_files(paths):
    """read and return the contents of paths"""
    docs = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as fp:
                print(p)
                docs.append(fp.read())
        except OSError:
            continue
    return docs


def load_encoder():
    """load the universal sentence encoder and cache it globally."""
    global ENCODER
    if ENCODER is None:
        url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        ENCODER = hub.load(url)
    return ENCODER
