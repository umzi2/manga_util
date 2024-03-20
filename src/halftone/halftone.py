import numpy as np
from pepeline import screentone


class Halftone:
    def __init__(self, dot_size: int):
        self.dot_size = dot_size
        pass

    def run(self, img: np.ndarray) -> np.ndarray:
        return screentone(img, self.dot_size)
