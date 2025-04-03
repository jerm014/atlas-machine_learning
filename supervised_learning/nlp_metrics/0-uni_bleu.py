#!/usr/bin/env python3
"""uni bleu"""
import math
from collections import Counter


def uni_bleu(references, sentence):
    """ calculates unigram BLEU score for a candidate sentence """
    # count unigrams in candidate sentence
    counts = Counter(sentence)

    # count maximum reference counts for each word
    max_ref_counts = {}
    for ref in references:
        ref_counts = Counter(ref)
        for word in ref_counts:
            max_ref_counts[word] = max(max_ref_counts.get(word, 0),
                                       ref_counts[word])

    # clipped count
    clipped_count = 0
    for word in counts:
        clipped_count += min(counts[word], max_ref_counts.get(word, 0))

    # precision
    total_count = len(sentence)
    precision = clipped_count / total_count if total_count > 0 else 0

    # calculate bp (brevity penalty)
    # c is length of the candidate sentence (i.e., the number of words in the
    #   proposed/generated sentence)
    # r is effective reference length (avoid unfairly penalizing shorter or
    #   longer candidate sentences)
    c = len(sentence)
    r = min((abs(len(ref) - c), len(ref)) for ref in references)[1]
    if c > r:
        bp = 1
    elif c == 0:
        bp = 0
    else:
        bp = math.exp(1 - r / c)

    # BLEU score
    bleu = bp * precision
    return bleu
