import argparse
from itertools import combinations
import os

from matplotlib import pyplot as plt
import cv2
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

import numpy as np

class BBox:
    """Class representing a single bounding box."""

    def __init__(self, x, y, w, h, roi):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.roi = roi
        self.histogram = self.calculate_histogram()


    def calculate_histogram(self):
        """Calculates histogram of the bounding box."""
        # https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
        hsv_roi = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)
        ranges = [0, 180, 0, 256,]
        h_bins = 50
        s_bins = 60
        hist_size = [h_bins, s_bins]
        hist_hsv_roi = cv2.calcHist([hsv_roi], [0, 1], None, hist_size, ranges, accumulate=False)
        cv2.normalize(hist_hsv_roi, hist_hsv_roi, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist_hsv_roi


class Frame:
    """Class representing a single frame."""

    def __init__(self, source_file, bbox_n, coordinates):
        self.source_file = source_file
        self.bbox_n = bbox_n
        self.coordinates = coordinates
        self.bboxes = []
        img = cv2.imread(source_file)
        for coord in coordinates:
            roi = img[int(coord[1]):int(coord[1]+coord[3]),
                      int(coord[0]):int(coord[0]+coord[2])]
            self.bboxes.append(BBox(*coord, roi))


class Matcher:
    """Class for matching bounding boxes between two frames."""

    def __init__(self, hyperparameters=None):
        self.G = FactorGraph()
        self.belief_propagation = None
        if hyperparameters is not None:
            self.LIKELY_MAX_DISTANCE = hyperparameters['likely_max_distance']
            self.HIST_SIMILARITY_THRESHOLD = hyperparameters['hist_similarity_threshold']
        else:
            self.LIKELY_MAX_DISTANCE = 120
            self.HIST_SIMILARITY_THRESHOLD = 0.25


    def set_frames(self, frame1, frame2):
        """Sets frames to be matched and resets the factor graph."""
        self.G = FactorGraph()
        self.frame1 = frame1
        self.frame2 = frame2
        self._add_variable_nodes()
        return self


    def add_histogram_comparission_factors(self):
        """Adds factors that compare histograms of bboxes."""
        for frame2_bbox_idx in range(self.frame2.bbox_n):
            factor_similarity_values = []
            for frame1_bbox_idx in range(self.frame1.bbox_n):
                similarity = cv2.compareHist(
                    self.frame1.bboxes[frame1_bbox_idx].histogram,
                    self.frame2.bboxes[frame2_bbox_idx].histogram,
                    cv2.HISTCMP_CORREL)
                factor_similarity_values.append(similarity)
            # 0.25 is the value that will be picked in case that none histogram is similar enough
            df = DiscreteFactor(
                [f'X_{frame2_bbox_idx}'],
                [self.frame1.bbox_n + 1],
                [[ self.HIST_SIMILARITY_THRESHOLD ] + factor_similarity_values ])
            self.G.add_factors(df)
            self.G.add_edge(f'X_{frame2_bbox_idx}', df)
        return self


    def add_duplication_avoidance_factors(self):
        """Adds factors that prohibit choosing the same bbox twice."""
        # Prohibit (by adding zeros on diagonal of the matrix) choosing the same bbox twice
        values = np.ones((self.frame1.bbox_n + 1, self.frame1.bbox_n + 1))
        np.fill_diagonal(values, 0)
        values[0, 0] = 1 # For the case where no bbox is chosen
        for i, j in combinations(range(self.frame2.bbox_n), 2):
            df = DiscreteFactor(
                [f'X_{i}', f'X_{j}'],
                [self.frame1.bbox_n + 1, self.frame1.bbox_n + 1],
                values)
            self.G.add_factors(df)
            self.G.add_edge(f'X_{i}', df)
            self.G.add_edge(f'X_{j}', df)
        return self


    def add_distance_factors(self):
        """Adds factors that penalize choosing bboxes that are too far from each other."""
        for frame2_bbox_idx in range(self.frame2.bbox_n):
            inv_distances = []
            for frame1_bbox_idx in range(self.frame1.bbox_n):
                box1 = self.frame1.bboxes[frame1_bbox_idx]
                box2 = self.frame2.bboxes[frame2_bbox_idx]
                box1_center = np.array([box1.x + box1.w / 2, box1.y + box1.h / 2])
                box2_center = np.array([box2.x + box2.w / 2, box2.y + box2.h / 2])
                distance = np.linalg.norm(box1_center - box2_center)
                if distance == 0:
                    inv_distance = 1 / 0.0001 # Not sure if this is not too extreme
                else:
                    inv_distance = 1 / distance
                inv_distances.append(inv_distance)
            # -1 will be picked if no box is closer than LIKELY_MAX_DISTANCE
            df = DiscreteFactor(
                [f'X_{frame2_bbox_idx}'],
                [self.frame1.bbox_n + 1],
                [[ 1 / self.LIKELY_MAX_DISTANCE ] + inv_distances])
            self.G.add_factors(df)
            self.G.add_edge(f'X_{frame2_bbox_idx}', df)
        return self


    def finish(self):
        """Finishes graph construction and calibrates belief propagation."""
        self.belief_propagation = BeliefPropagation(self.G)
        self.belief_propagation.calibrate()
        return self


    def match(self):
        """Inference"""
        matching = self.belief_propagation.map_query(self.G.get_variable_nodes())
        matching.update((key, value - 1) for key, value in matching.items())
        return dict(sorted(matching.items()))


    def _add_variable_nodes(self):
        """Adds variable nodes from second frame to the graph."""
        for index in range(self.frame2.bbox_n):
            self.G.add_node(f'X_{index}')


def parse_labels(root_path, with_ground_truth=False):
    """Utility function parsing label files into Frame objects."""
    target = 'bboxes.txt' if not with_ground_truth else 'bboxes_gt.txt'
    with open(os.path.join(root_path, target), 'r') as f:
        lines = f.readlines()
    frames = []
    per_frame_ground_truths = []
    while len(lines) > 0:
        source_file = lines.pop(0).strip()
        bbox_n = int(lines.pop(0).strip())
        coordinates = []
        ground_truths = []
        for _ in range(bbox_n):
            if with_ground_truth:
                ground_truth, *coords = lines.pop(0).strip().split()
                ground_truths.append(ground_truth)
                coordinates.append(coords)
            else:
                coordinates.append(lines.pop(0).strip().split())
        coordinates = [[float(coord) for coord in coordinate] for coordinate in coordinates]
        frames.append(Frame(os.path.join(root_path, 'frames', source_file), bbox_n, coordinates))
        if with_ground_truth:
            per_frame_ground_truths.append(ground_truths)
    return frames, per_frame_ground_truths


def accuracy_metric(matching, ground_truths):
    """Simple metric calculating number of correct matches and incorrect matches."""
    correct = sum([1 for match, ground_truth in zip(matching, ground_truths) if match == ground_truth])
    incorrect = len(matching) - correct
    return correct, incorrect


# Argument
parser = argparse.ArgumentParser(
    description="Inference script for bbox index matching between frames")
parser.add_argument('dataset_root', type=str, help="Path to dataset root directory")
parser.add_argument('--with_ground_truth', action='store_true', help="Whether to use ground truth for evaluation")
parser.add_argument('--verbose', action='store_true', help="Whether to print verbose output")


if __name__ == '__main__':
    dataset_root = parser.parse_args().dataset_root
    with_ground_truth = parser.parse_args().with_ground_truth

    if parser.parse_args().verbose:
        print(f"Loading dataset from root directory: {dataset_root}...")

    frames, ground_truths = parse_labels(os.path.join(dataset_root), with_ground_truth)

    correct_cumulative = 0
    incorrect_cumulative = 0
    matcher = Matcher()

    if parser.parse_args().verbose:
        print("Inference...")
        print(f"Hyperparameters: LIKELY_MAX_DISTANCE: {matcher.LIKELY_MAX_DISTANCE}, HIST_SIMILARITY_THRESHOLD: {matcher.HIST_SIMILARITY_THRESHOLD}")

    # Main inference loop
    for i in range(len(frames)):
        if i == 0:
            print(('-1 ' * frames[i].bbox_n).strip())
            continue
        if frames[i].bbox_n == 0:
            print()
            continue
        if frames[i - 1].bbox_n == 0:
            print(('-1 ' * frames[i].bbox_n).strip())
            continue

        matcher.set_frames(frames[i - 1], frames[i]) \
            .add_histogram_comparission_factors() \
            .add_duplication_avoidance_factors() \
            .add_distance_factors() \
            .finish()
        matching = matcher.match()

        # Accuracy calculation
        if with_ground_truth:
            correct, incorrect = accuracy_metric(matching.values(), [int(item) for item in ground_truths[i]])
            correct_cumulative += correct
            incorrect_cumulative += incorrect

        # Preparing and displaying display view
        matching_output_view = list(matching.values())
        matching_output_view = ' '.join([str(item) for item in matching_output_view])
        print(f"{matching_output_view}")

        if parser.parse_args().verbose:
            print(f"matching: {matching}, ground_truth: {ground_truths[i]}")
            print(f"Single sample accuracy: {correct} / {correct + incorrect} = {correct / (correct + incorrect)}")

    if with_ground_truth and parser.parse_args().verbose:
        print(f"Total accuracy: {correct_cumulative} / {correct_cumulative + incorrect_cumulative} = {correct_cumulative / (correct_cumulative + incorrect_cumulative)}")
