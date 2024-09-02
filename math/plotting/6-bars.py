#!/usr/bin/env python3
"""Module to create a stacked bar graph of fruit quantities per person."""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Create and display a stacked bar graph of fruit quantities per person.

    The graph shows the number of apples, bananas, oranges, and peaches
    possessed by Farrah, Fred, and Felicia.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
    fruit_colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    bottom = np.zeros(3)  # Initialize the bottom of each stack
    for i, fruit_type in enumerate(fruit):
        plt.bar(people, fruit_type, 0.5, bottom=bottom,
                label=fruit_names[i], color=fruit_colors[i])
        bottom += fruit_type  # Update the bottom for the next stack

    # Customize the graph
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.legend(loc='upper right')

    # Set y-axis range and ticks
    plt.ylim(0, 80)
    plt.yticks(range(0, 81, 10))

    # Display the plot
    plt.show()
