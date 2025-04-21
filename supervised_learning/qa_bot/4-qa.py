#!/usr/bin/env python3
"""
multi‑document question‑answer interface for task 4

this module exposes question_answer(folder) that loads every text document in
the directory and then

   prompts for a question with Q:
   picks the document most semantically similar to the query using
       semantic_search from the 3‑semantic_search module
   extracts an answer span from that document with question_answer from task 0
   prints the answer as A:

end by typing exit quit goodbye bye (case insensitive)
"""
import os
import sys
semantic_search = __import__("3-semantic_search").semantic_search
# note: this question_answer needs (question, reference)
extract_answer = __import__("0-qa").question_answer
EXIT_KEYWORDS = {"exit", "quit", "goodbye", "bye"}


def question_answer(coprus_path):
    """
    run an interactive question‑answer loop over a corpus directory

    Args:
        coprus_path: directory containing plain‑text reference documents.
    """
    # validate path early
    if not os.path.isdir(coprus_path):
        sys.exit(f"error: '{coprus_path}' is not a directory")

    print("ask your questions! (type 'exit' to quit)")

    while True:
        try:
            query = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            query = "exit"

        if query.lower() in EXIT_KEYWORDS:
            print("A: Goodbye")
            break

        if not query:
            continue

        # select most relevant document
        reference = semantic_search(coprus_path, query)
        if reference is None:
            print("A: Sorry, I do not have any reference documents.")
            continue

        # don't be confused. this calls question_answer from 0-qa
        answer = extract_answer(query, reference)
        if answer is None or not answer:
            answer = "Sorry, I do not know."
        print(f"A: {answer}")
