#!/usr/bin/env python3
""" do a line thing in this file. """
import numpy as np
import matplotlib.pyplot as plt


def line():
    """ make a line thing """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(np.arange(0, 11), y, color='red', linestyle='-')
    plt.xlim(0, 10)
    plt.show()
