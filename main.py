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
        for i in range(self.frame2.bbox_n):
            for j in range(self.frame1.bbox_n):
                self.G.add_factor([f'X_{i}', f'X_{j}'], self.frame2.bboxes[i].calculate_histogram(), self.frame1.bboxes[j].calculate_histogram())
        return self


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
    plt.plot(frames[0].bboxes[0].calculate_histogram())
    print(frames[0].bboxes[0].calculate_histogram().shape)
    plt.show()