#!/usr/bin/env python3
"""Module that defines a single neuron performing binary classification"""
import cv2
import numpy as np

# Load the image
img = cv2.imread('cat_damaged.png', cv2.IMREAD_COLOR)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find all the BLACK pixels to create a mask
mask = cv2.inRange(gray, 0, 1)
mask = cv2.bitwise_not(mask)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    mask, connectivity=8)

cleaned_mask = np.zeros_like(mask)
# Skip label 0 as it's the background
for label in range(1, num_labels):
    size = stats[label, cv2.CC_STAT_AREA]
    if size > 50:  # Only keep components larger than 50 pixels
        cleaned_mask[labels == label] = 255

#save the mask to a file
cv2.imwrite('mask.png', cleaned_mask)

