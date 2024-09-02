#!/usr/bin/env python3
""" two file for task 3 """
import numpy as np
import matplotlib.pyplot as plt


def two():
    """ two function makes a graph. """

    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plot x y1 with a dashed red line
    plt.plot(x, y1, 'r--', label='C-14')

    # Plot x y2 with a solid green line
    plt.plot(x, y2, 'g-', label='Ra-226')

    # Set labels for x-axis and y-axis
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')

    # Set the title
    plt.title('Exponential Decay of Radioactive Elements')

    # Set x-axis range
    plt.xlim(0, 20000)

    # Set y-axis range
    plt.ylim(0, 1)

    # Add legend in the upper right corner
    plt.legend(loc='upper right')

    # Display the plot
    plt.show()
