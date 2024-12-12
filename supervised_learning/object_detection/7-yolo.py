#!/usr/bin/env python3
"""
Module for YOLO v3 object detection implementation.
This module provides functionality for object detection using YOLO v3
algorithm with Darknet and Keras.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os


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
            filtered_boxes:  numpy.ndarray of shape (?, 4) containing
                             filtered bounding boxes
            box_classes:     numpy.ndarray of shape (?,) containing class
                             number for each filtered box
            box_scores:      numpy.ndarray of shape (?) containing box scores
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

    def intersection_over_union(self, box1, boxes):
        """
        Calculate intersection over union between boxes.

        Args:
            box1:  numpy.ndarray of shape (4,) containing first box
                   coordinates
            boxes: numpy.ndarray of shape (n, 4) containing n box
                   coordinates

        Returns:
            numpy.ndarray of shape (n,) containing IoU values for each box
        """
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])

        intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union_area = box1_area + boxes_area - intersection_area

        return intersection_area / union_area

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Perform non-max suppression on filtered boxes.

        Args:
           filtered_boxes:  numpy.ndarray of shape (?, 4) containing
                            filtered bounding boxes
           box_classes:     numpy.ndarray of shape (?,) containing class
                            numbers for filtered_boxes
           box_scores:      numpy.ndarray of shape (?) containing box scores
                            for filtered_boxes

        Returns:
           box_predictions:       numpy.ndarray of shape (?, 4) containing
                                  predicted boxes ordered by class and score
           predicted_box_classes: numpy.ndarray of shape (?,) containing
                                  class numbers ordered by class and score
           predicted_box_scores:  numpy.ndarray of shape (?) containing box
                                  scores ordered by class and score
        """
        if len(filtered_boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        # sort by box classes, group the box classes together, then within
        # each class, sort by box scores in descending order
        idxs = np.lexsort((-box_scores, box_classes))

        box_predictions = filtered_boxes[idxs]
        predicted_box_classes = box_classes[idxs]
        predicted_box_scores = box_scores[idxs]

        selected_idxs = []
        # np.unique() returns the sorted unique elements
        unique_classes = np.unique(predicted_box_classes)
        # only iterate through classes that actually have detected boxes
        for cls in unique_classes:
            class_mask = predicted_box_classes == cls
            class_idxs = np.where(class_mask)[0]

            while len(class_idxs) > 0:
                selected_idxs.append(class_idxs[0])

                if len(class_idxs) == 1:
                    break

                ious = self.intersection_over_union(
                    box_predictions[class_idxs[0]],
                    box_predictions[class_idxs[1:]]
                )

                class_idxs = class_idxs[1:][ious < self.nms_t]

        selected_idxs = np.array(selected_idxs)

        return (box_predictions[selected_idxs],
                predicted_box_classes[selected_idxs],
                predicted_box_scores[selected_idxs])

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Display image with bounding boxes, class names, and scores.

        Args:
            image:       numpy.ndarray containing unprocessed image
            boxes:       numpy.ndarray containing boundary boxes coordinates
            box_classes: numpy.ndarray containing class indices for each box
            box_scores:  numpy.ndarray containing box scores
            file_name:   file path of the original image

        Returns:
            None
        """
        # Create a copy of the image to draw on
        img_copy = image.copy()

        for i, box in enumerate(boxes):
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, box)

            # Draw box with blue color
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Create text with class name and score
            class_name = self.class_names[box_classes[i]]
            score = f"{box_scores[i]:.2f}"
            text = f"{class_name} {score}"

            # Get text size for positioning
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )

            # Draw text background and text
            text_x = x1
            text_y = max(y1 - 5, text_height)  # Ensure text stays in image

            cv2.putText(
                img_copy, text, (text_x, text_y),
                font, font_scale, (0, 0, 255),
                thickness, cv2.LINE_AA
            )

        # Display image
        cv2.imshow(file_name, img_copy)

        # Wait for key press
        key = cv2.waitKey(0)

        # If 's' is pressed, save the image
        if key == ord('s'):
            # Create detections directory if it doesn't exist
            if not os.path.exists('detections'):
                os.makedirs('detections')

            # Save image
            output_path = os.path.join('detections', file_name)
            cv2.imwrite(output_path, img_copy)

        # Close window
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Perform object detection on all images in specified folder.

        Args:
            folder_path: path to folder containing images to process

        Returns:
            tuple containing:
                - predictions: List of tuples for each image containing:
                               (boxes, box_classes, box_scores)
                - image_paths: List of str containing paths to each image

        Raises:
            ValueError: If folder_path is invalid or contains no images
        """
        # Load all images from folder
        images, image_paths = self.load_images(folder_path)

        # Preprocess images
        pimages, image_shapes = self.preprocess_images(images)

        # Get predictions for all images
        predictions = []
        outputs = self.model.predict(pimages)

        # Process predictions for each image
        for i, img in enumerate(images):
            # Extract outputs for current image
            img_outputs = [
                output[i:i + 1] if len(output.shape) == 4
                else output[i] for output in outputs
            ]

            # Process outputs to get boxes and scores
            boxes, confidences, class_probs = self.process_outputs(
                img_outputs, image_shapes[i]
            )

            # Filter boxes based on class threshold
            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes, confidences, class_probs
            )

            # Apply non-max suppression
            box_pred, pred_classes, pred_scores = self.non_max_suppression(
                filtered_boxes, box_classes, box_scores
            )

            # Add to predictions list
            predictions.append(
                (box_pred, pred_classes, pred_scores)
            )

            # Display results
            # Get filename without path for window name
            file_name = os.path.basename(image_paths[i])
            self.show_boxes(
                img, box_pred, pred_classes, pred_scores, file_name
            )

        return predictions, image_paths

    def preprocess_images(self, images):
        """
        Preprocess images for inference with Darknet model.

        Args:
           images: List of numpy.ndarray containing image data

        Returns:
           tuple containing:
               - pimages: numpy.ndarray of shape (ni, input_h, input_w, 3)
                         containing preprocessed images
               - image_shapes: numpy.ndarray of shape (ni, 2) containing
                             original image dimensions

        Raises:
           ValueError: If images list is empty
        """

        if not images:
            raise ValueError("Empty images list provided")

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            image_shapes.append(img.shape[:2])
            resized = cv2.resize(
                img,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )
            pimages.append(resized / 255)

        pimages = np.array(pimages).reshape(-1, input_h, input_w, 3)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    @staticmethod
    def load_images(folder_path):
        """
        Load all images from specified folder path.

        Args:
            folder_path: Path to folder containing images to load

        Returns:
            tuple containing:
                - images: List of numpy.ndarray containing loaded images
                - image_paths: List of image file paths

        Raises:
            ValueError: If folder_path does not exist or contains no images
        """
        if not os.path.exists(folder_path):
            raise ValueError(f"Path {folder_path} does not exist")

        image_paths = []
        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                image_paths.append(os.path.join(folder_path, filename))

        if not image_paths:
            raise ValueError(f"No images found in {folder_path}")

        images = [cv2.imread(path) for path in image_paths]

        return images, image_paths


def sigmoid(x):
    """Apply sigmoid activation function."""
    return 1 / (1 + np.exp(-x))
