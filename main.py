import os
import sys
from PIL import Image
from pgmpy.models import FactorGraph

class BBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class Frame:
    def __init__(self, source_file, bbox_n, coordinates):
        self.source_file = source_file
        self.bbox_n = bbox_n
        self.coordinates = coordinates
        self.bboxes = []
        for coord in coordinates:
            self.bboxes.append(BBox(*coord))


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


    def _add_variable_nodes(self):
        for index in range(self.frame1.bbox_n):
            self.G.add_node(f'X_{index}')


def parse_labels(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    frames = []
    while len(lines) > 0:
        source_file = lines.pop(0).strip()
        bbox_n = int(lines.pop(0).strip())
        coordinates = []
        for _ in range(bbox_n):
            coordinates.append(lines.pop(0).strip().split())
        frames.append(Frame(source_file, bbox_n, coordinates))
    return frames


if __name__ == '__main__':
    dataset_path = sys.argv[1]
    print(f"Loading dataset from: {dataset_path}...")
    parse_labels(dataset_path)
    frames = parse_labels(dataset_path)
    print(frames[1].coordinates)
    matcher = Matcher(frames[0], frames[1])
    matcher.set_frames(frames[1], frames[2])
    print(matcher.G.nodes)