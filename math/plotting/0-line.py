#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    
    # Plot y as a solid red line
    plt.plot(np.arange(0, 11), y, color='red', linestyle='-')

    # Set x-axis range from 0 to 10
    plt.xlim(0, 10)

    # Add labels and title (optional, but recommended for clarity)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Cubic Function: y = x^3')

    # Display the plot
    plt.show()
