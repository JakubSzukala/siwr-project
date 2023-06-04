import argparse
from itertools import combinations
import os
import sys

from matplotlib import pyplot as plt
import cv2
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

import numpy as np

class BBox:
    def __init__(self, x, y, w, h, roi):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.roi = roi
        self.histogram = self.calculate_histogram()


    def calculate_histogram(self):
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
    def __init__(self):
        self.G = FactorGraph()
        self.belief_propagation = None


    def set_frames(self, frame1, frame2):
        self.G = FactorGraph()
        self.frame1 = frame1
        self.frame2 = frame2
        self._add_variable_nodes()
        return self


    def add_histogram_comparission_factors(self):
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
                [[ 0.25 ] + factor_similarity_values ])
            self.G.add_factors(df)
            self.G.add_edge(f'X_{frame2_bbox_idx}', df)
        return self


    def add_duplication_avoidance_factors(self):
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


    def finish(self):
        self.belief_propagation = BeliefPropagation(self.G)
        self.belief_propagation.calibrate()
        return self


    def match(self):
        matching = self.belief_propagation.map_query(self.G.get_variable_nodes())
        matching.update((key, value - 1) for key, value in matching.items())
        return dict(sorted(matching.items()))


    def _add_variable_nodes(self):
        for index in range(self.frame2.bbox_n):
            self.G.add_node(f'X_{index}')


def parse_labels(root_path, with_ground_truth=False):
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
    return sum([1 for match, ground_truth in zip(matching, ground_truths) if match == ground_truth]) / len(matching)


parser = argparse.ArgumentParser(
    description="Inference script for bbox index matching between frames")
parser.add_argument('dataset_root', type=str, help="Path to dataset root directory")
parser.add_argument('--with_ground_truth', action='store_true', help="Whether to use ground truth for evaluation")


if __name__ == '__main__':
    dataset_root = parser.parse_args().dataset_root
    with_ground_truth = parser.parse_args().with_ground_truth
    print(f"Loading dataset from root directory: {dataset_root}...")
    frames, ground_truths = parse_labels(os.path.join(dataset_root), with_ground_truth)

    matcher = Matcher()
    for i in range(1, len(frames)):
        matcher.set_frames(frames[i - 1], frames[i]) \
            .add_histogram_comparission_factors() \
            .add_duplication_avoidance_factors() \
            .finish()
        matching = matcher.match()
        print(f"matching: {matching}, ground_truth: {ground_truths[i]}")
        accuracy = accuracy_metric(matching.values(), [int(item) for item in ground_truths[i]])
        print(f"Accuracy: {accuracy}")
        input()