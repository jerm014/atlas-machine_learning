#!/usr/bin/env python3
"""
Module for YOLO v3 object detection implementation.
This module provides functionality for object detection using YOLO v3
algorithm with Darknet and Keras.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


class Yolo:
    """
    A class to perform object detection using YOLO v3 algorithm.

    This class implements the YOLO v3 object detection algorithm using
    a pre-trained Darknet model converted to Keras format.

    Attributes:
        model:       Darknet Keras model
        class_names: List of class names for the model
        class_t:     Box score threshold for initial filtering
        nms_t:       IOU threshold for non-max suppression
        anchors:     Anchor boxes for predictions
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize the Yolo class.

        Args:
            model_path:   Path to the Darknet Keras model
            classes_path: Path to the list of class names
            class_t:      Box score threshold for initial filtering
            nms_t:        IOU threshold for non-max suppression
            anchors:      Numpy array of shape (outputs, anchor_boxes, 2)
                          containing anchor boxes

        Returns:
            None
        """
        try:
            self.model = keras.models.load_model(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")

        try:
            with open(classes_path, 'r') as f:
                self.class_names = [line.strip() for line in f]
        except Exception as e:
            raise ValueError(f"Failed to load classes from {classes_path}: " +
                             "{e}")

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

        # Validate input parameters
        if not isinstance(class_t, float) or not 0 <= class_t <= 1:
            raise ValueError("class_t must be a float between 0 and 1")

        if not isinstance(nms_t, float) or not 0 <= nms_t <= 1:
            raise ValueError("nms_t must be a float between 0 and 1")

        if not isinstance(anchors, np.ndarray) or len(anchors.shape) != 3:
            raise ValueError(
                "anchors must be a numpy.ndarray of shape (outputs, " +
                "anchor_boxes, 2)"
            )

        if anchors.shape[2] != 2:
            raise ValueError(
                "Last dimension of anchors must be 2 [width, height]"
            )

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs from the Darknet model.

        Args:
            outputs:    List of numpy.ndarrays containing predictions from
                        the Darknet model for a single image
            image_size: numpy.ndarray containing original image size
                        [image_height, image_width]

        Returns:
            boxes: List of numpy.ndarrays of processed boundary boxes
            box_confidences: List of numpy.ndarrays of box confidences
            box_class_probs: List of numpy.ndarrays of class probabilities
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width = output.shape[:2]
            box_conf = output[..., 4:5]
            box_class_prob = output[..., 5:]

            box_confidences.append(sigmoid(box_conf))
            box_class_probs.append(sigmoid(box_class_prob))

            box_xy = sigmoid(output[..., :2])
            box_wh = np.exp(output[..., 2:4])
            anchors_tensor = self.anchors[i]

            col = np.tile(
                np.arange(0, grid_width), grid_height
            ).reshape(grid_height, grid_width)
            row = np.tile(
                np.arange(0, grid_height).reshape(-1, 1), grid_width
            )

            col = col.reshape(grid_height, grid_width, 1, 1).repeat(
                self.anchors.shape[1], axis=2
            )
            row = row.reshape(grid_height, grid_width, 1, 1).repeat(
                self.anchors.shape[1], axis=2
            )
            grid = np.concatenate((col, row), axis=3)

            box_xy += grid
            box_xy /= (grid_width, grid_height)
            box_wh *= anchors_tensor
            box_wh /= self.model.input.shape[1:3]

            box_x1y1 = box_xy - (box_wh / 2)
            box_x2y2 = box_xy + (box_wh / 2)
            box = np.concatenate((box_x1y1, box_x2y2), axis=-1)

            box *= np.tile(image_size, 2)

            boxes.append(box)

        return boxes, box_confidences, box_class_probs


def sigmoid(x):
    """Apply sigmoid activation function."""
    return 1 / (1 + np.exp(-x))
