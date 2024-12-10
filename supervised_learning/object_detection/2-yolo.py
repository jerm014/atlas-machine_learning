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
            outputs: List of numpy.ndarrays containing predictions from the
                    Darknet model for a single image
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
        image_height, image_width = image_size

        for idx, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            box_conf = output[..., 4:5]
            box_class_prob = output[..., 5:]
            box_confidences.append(sigmoid(box_conf))
            box_class_probs.append(sigmoid(box_class_prob))

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            grid_x = np.arange(grid_width).reshape(1, grid_width, 1)
            grid_x = np.repeat(grid_x, grid_height, axis=0)
            grid_x = np.repeat(grid_x, anchor_boxes, axis=2)

            grid_y = np.arange(grid_height).reshape(grid_height, 1, 1)
            grid_y = np.repeat(grid_y, grid_width, axis=1)
            grid_y = np.repeat(grid_y, anchor_boxes, axis=2)

            bx = (sigmoid(tx) + grid_x) / grid_width
            by = (sigmoid(ty) + grid_y) / grid_height

            pw = self.anchors[idx, :, 0]
            ph = self.anchors[idx, :, 1]

            bw = pw * np.exp(tw) / self.model.input.shape[1]
            bh = ph * np.exp(th) / self.model.input.shape[2]

            x1 = (bx - (bw / 2)) * image_width
            y1 = (by - (bh / 2)) * image_height
            x2 = (bx + (bw / 2)) * image_width
            y2 = (by + (bh / 2)) * image_height

            box = np.zeros(output[..., :4].shape)
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2
            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter YOLO boxes based on class and object confidence.

        Args:
            boxes:           List of numpy.ndarrays of shape
                             (grid_height, grid_width, anchor_boxes, 4)
                             containing processed boundary boxes
            box_confidences: List of numpy.ndarrays of shape
                             (grid_height, grid_width, anchor_boxes, 1)
                             containing processed box confidences
            box_class_probs: List of numpy.ndarrays of shape
                             (grid_height, grid_width, anchor_boxes, classes)
                             containing processed box class probabilities

        Returns:
            filtered_boxes: numpy.ndarray of shape (?, 4) containing
                           filtered bounding boxes
            box_classes: numpy.ndarray of shape (?,) containing class number
                        for each filtered box
            box_scores: numpy.ndarray of shape (?) containing box scores
                       for each filtered box
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box_scores_per_class = box_confidences[i] * box_class_probs[i]
            box_class = np.argmax(box_scores_per_class, axis=-1)
            box_score = np.max(box_scores_per_class, axis=-1)
            mask = box_score >= self.class_t

            filtered_boxes.extend(boxes[i][mask])
            box_classes.extend(box_class[mask])
            box_scores.extend(box_score[mask])

        filtered_boxes = np.array(filtered_boxes)
        box_classes = np.array(box_classes)
        box_scores = np.array(box_scores)

        return filtered_boxes, box_classes, box_scores


def sigmoid(x):
    """Apply sigmoid activation function."""
    return 1 / (1 + np.exp(-x))
