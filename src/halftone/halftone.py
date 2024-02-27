import numpy as np
from screenton_maker import Screenton


class Halftone:
    def __init__(self, dot_size):
        self.dot_size = dot_size
        pass

    def run(self, img):
        dot = Screenton(self.dot_size)
        return dot.run(img.astype(np.float32))
