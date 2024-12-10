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
        model: Darknet Keras model
        class_names: List of class names for the model
        class_t: Box score threshold for initial filtering
        nms_t: IOU threshold for non-max suppression
        anchors: Anchor boxes for predictions
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize the Yolo class.

        Args:
            model_path: Path to the Darknet Keras model
            classes_path: Path to the list of class names
            class_t: Box score threshold for initial filtering
            nms_t: IOU threshold for non-max suppression
            anchors: Numpy array of shape (outputs, anchor_boxes, 2)
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
