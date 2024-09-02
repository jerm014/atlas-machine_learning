#!/usr/bin/env python3
"""Module to create a stacked bar graph of fruit quantities per person."""
import numpy as np
import matplotlib.pyplot as plt


def stacked_bar():
    """
    Create and display a stacked bar graph of fruit quantities per person.

    The graph shows the number of apples, bananas, oranges, and peaches
    possessed by Farrah, Fred, and Felicia.
    """
    # Define the data
    fruit = np.array([[30, 25, 22],   # apples
                      [40, 32, 28],   # bananas
                      [20, 15, 18],   # oranges
                      [15, 13, 12]])  # peaches
    people = ['Farrah', 'Fred', 'Felicia']
    fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
    fruit_colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    # Create the stacked bar graph
    fig, ax = plt.subplots(figsize=(8, 6))

    bottom = np.zeros(3)  # Initialize the bottom of each stack
    for i, fruit_type in enumerate(fruit):
        ax.bar(people, fruit_type, 0.5, bottom=bottom,
               label=fruit_names[i], color=fruit_colors[i])
        bottom += fruit_type  # Update the bottom for the next stack

    # Customize the graph
    ax.set_ylabel('Quantity of Fruit')
    ax.set_title('Number of Fruit per Person')
    ax.legend(loc='upper right')

    # Set y-axis range and ticks
    ax.set_ylim(0, 80)
    ax.set_yticks(range(0, 81, 10))

    # Display the plot
    plt.show()
