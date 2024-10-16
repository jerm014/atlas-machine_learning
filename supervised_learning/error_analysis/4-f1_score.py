#!/usr/bin/env python3
"""f1 function for project 2295 task 4"""

import numpy as np


def f1_score(confusion):
    """
    I would love to write more awesome documenation
    but I'm kind of out of time.
    """
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision

    s = sensitivity(confusion)
    p = precision(confusion)

    f1 = 2 * (p * s) / (p + s)
    
    return f1
