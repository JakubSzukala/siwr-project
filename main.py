import os
import sys

from matplotlib import pyplot as plt
import cv2
from pgmpy.models import FactorGraph

class BBox:
    def __init__(self, x, y, w, h, roi):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.roi = roi


    def calculate_histogram(self):
        per_channel_hist = []
        for channel in cv2.split(self.roi):
            per_channel_hist.append(cv2.calcHist([channel], [0], None, [256], [0, 256]))
        return per_channel_hist


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
    def __init__(self, frame1, frame2):
        self.frame1 = frame1
        self.frame2 = frame2
        self.G = FactorGraph()
        self._add_variable_nodes()


    def set_frames(self, frame1, frame2):
        self.frame1 = frame1
        self.frame2 = frame2
        self._add_variable_nodes()
        return self


    def add_histogram_factors(self):
        pass


    def _add_variable_nodes(self):
        for index in range(self.frame1.bbox_n):
            self.G.add_node(f'X_{index}')


def parse_labels(root_path):
    with open(os.path.join(root_path, 'bboxes.txt'), 'r') as f:
        lines = f.readlines()
    frames = []
    while len(lines) > 0:
        source_file = lines.pop(0).strip()
        bbox_n = int(lines.pop(0).strip())
        coordinates = []
        for _ in range(bbox_n):
            coordinates.append(lines.pop(0).strip().split())
            coordinates = [[float(coord) for coord in coordinate] for coordinate in coordinates]
        frames.append(Frame(os.path.join(root_path, 'frames', source_file), bbox_n, coordinates))
    return frames


if __name__ == '__main__':
    dataset_root = sys.argv[1]
    print(f"Loading dataset from root directory: {dataset_root}...")
    frames = parse_labels(os.path.join(dataset_root))
    print(frames[1].coordinates)
    matcher = Matcher(frames[0], frames[1])
    matcher.set_frames(frames[1], frames[2])
    print(matcher.G.nodes)
    frames[0].bboxes[0].calculate_histogram()
    cv2.imshow('roi', frames[0].bboxes[0].roi)
    cv2.waitKey(0)
    plt.plot(frames[0].bboxes[0].calculate_histogram()[0])
    plt.show()