#!/usr/bin/env python3
"""Module containing moving_average function"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.
    Args:
        data (list):  List of data points.
        beta (float): Weight factor for the moving average.
    Returns:
        list:         List of moving averages.
    """
    v = 0
    moving_averages = []
    for i, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x
        moving_averages.append(v / (1 - beta**i))
    return moving_averages
