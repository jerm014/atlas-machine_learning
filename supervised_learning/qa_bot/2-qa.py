#!/usr/bin/env python3
"""
TASK 2 (loop! but also answer questions.)

usage:
    python3 2_qa.py <reference1> [<reference2> ...]

note: this is also task one because I misunderstood task one and did too much
"""
import sys
question_answer = __import__("0-qa").question_answer
EXIT_KEYWORDS = {"exit", "quit", "goodbye", "bye"}


def load_references(paths):
    """read each path and join contents separated by a blank line"""
    docs = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as fp:
                docs.append(fp.read())
        except OSError as exc:
            sys.exit(f"error: could not read '{p}': {exc}")
    return "\n\n".join(docs)


def answer_loop(reference):
    """
    run an interactive question‑answer loop using reference text

    the loop continues until the user types an exit keyword (global variable
    EXIT_KEYWORDS)
    answers are printed with the prefix A: if the model
    returns None just apologize.
    """

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            question = "exit"

        if question.lower() in EXIT_KEYWORDS:
            print("A: Goodbye")
            break

        if not question:
            continue

        answer = question_answer(question, reference)
        if answer is None:
            answer = "Sorry, I do not know."
        print(f"A: {answer}")


def _main():
    """command‑line entry point."""
    if len(sys.argv) < 2:
        sys.exit("usage: 1_chat.py <reference1> [<reference2> ...]")

    reference = load_references(sys.argv[1:])
    answer_loop(reference)


if __name__ == "__main__":
    _main()
