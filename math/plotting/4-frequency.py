#!/usr/bin/env python3
""" frequency stuff task. """
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """ show frequency graph """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Create histogram
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black',
             linewidth=1)

    # Set those labels for x-axis and y-axis
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')

    # Set some title
    plt.title('Project A')

    # Set x-axis ticks to show bins every 10 units
    plt.xticks(range(0, 101, 10))

    # Display the plot
    plt.show()
