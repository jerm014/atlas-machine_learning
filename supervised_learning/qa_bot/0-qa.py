#!/usr/bin/env python3
"""
TASK 0
question‑answering utility using bert (tensorflow 2)

this module exposes a single function, question_answer, that extracts an
answer span from a reference text given a natural language question. it
combines the bert‑uncased‑tf2‑qa model from tensorflow‑hub for inference
with the squad‑fine‑tuned
bert‑large‑uncased‑whole‑word‑masking‑finetuned‑squad tokenizer from
transformers for preprocessing.

example: (from 0-main.py)

from 0_qa import question_answer
answer = question_answer("When are PLDs?", reference_text)

expected output:

on - site days from 9 : 00 am to 3 : 00 pm
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """find an answer snippet in reference that responds to question.

    Args:
        question: the natural language question.
        reference: the reference document (context) in which to search.

    Returns:
        str or None
            a text span from reference that answers question, or None if no
            answer is confidently found.
    """
    tokenizer = load_tokenizer()
    model = load_model()

    inputs, encoded = prepare_inputs(question, reference, tokenizer)
    outputs = model(inputs)
    start_logits, end_logits = outputs[0], outputs[1]

    return extract_answer(start_logits, end_logits, encoded, tokenizer)


def load_tokenizer():
    """load and cache the pretrained bert tokenizer."""
    return BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )


def load_model():
    """load and cache the tensorflow‑hub qa model."""
    return hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")


def prepare_inputs(question, reference, tokenizer):
    """tokenize question and reference and format inputs for the model.

    Args:
        question: question text.
        reference: context text.
        tokenizer: bert tokenizer instance.

    Returns:
        dict[str, tf.Tensor],
        transformers.tokenization_utils_base.BatchEncoding
            inputs dict for the model and the full encdoed batch
    """
    encoded = tokenizer.encode_plus(
        question,
        reference,
        add_special_tokens=True,
        max_length=512,
        truncation="only_second",
        return_attention_mask=True,
        return_tensors="tf",
    )

    input_word_ids = encoded["input_ids"]
    input_mask = encoded["attention_mask"]
    input_type_ids = encoded["token_type_ids"]

    # the tf‑hub model expects a list
    # [word ids, attention mask, token type ids]
    inputs = [input_word_ids, input_mask, input_type_ids]
    return inputs, encoded


def extract_answer(start_logits, end_logits, encoded, tokenizer):
    """
    this convert model logits to the best answer span or None

    Args:
        start_logits: Tensor with start indices scores.
        end_logits: Tensor with end indices scores
        encoded: tokenized batch from prepare_inputs.
        tokenizer: bert tokenizer instance

    Returns:
        answer as a string or None
    """
    start = start_logits.numpy().squeeze()
    end = end_logits.numpy().squeeze()

    # mask out positions that belong to the question (segment id 0) so that
    # only reference tokens (segment id 1) can be chosen for the answer span
    token_type_ids = encoded["token_type_ids"][0].numpy()
    ref_mask = token_type_ids == 1

    # build a 2‑d mask where a span is valid only if both endpoints are in the
    # reference segment and end >= start
    n_tokens = start.shape[0]
    span_scores = np.full((n_tokens, n_tokens), -np.inf, dtype=np.float32)

    for i in range(n_tokens):
        if not ref_mask[i]:
            continue
        for j in range(i, min(i + 30, n_tokens)):  # limit span length <= 30
            if ref_mask[j]:
                span_scores[i, j] = start[i] + end[j]

    if np.all(np.isneginf(span_scores)):
        return None

    start_idx, end_idx = np.unravel_index(
        np.argmax(span_scores), span_scores.shape)

    tokens = encoded["input_ids"][0][start_idx: end_idx + 1]
    answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()

    if answer:
        return answer
    else:
        return None
