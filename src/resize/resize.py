import numpy as np
from chainner_ext import resize, ResizeFilter


class Resize:

    def __init__(self, size: int, interpolation: str, width: bool, percent: int, spread: bool, spread_size: int):

        self.size = size
        self.interpolation = interpolation
        self.width = width
        self.percent = percent / 100
        self.spread = spread
        self.spread_size = spread_size

    def run(self, img: np.ndarray) -> np.ndarray:
        interpolation_map = {
            'nearest': ResizeFilter.Nearest,
            'linear': ResizeFilter.Linear,
            'cubic_catrom': ResizeFilter.CubicCatrom,
            'cubic_mitchell': ResizeFilter.CubicMitchell,
            'cubic_bspline': ResizeFilter.CubicBSpline,
            'lanczos': ResizeFilter.Lanczos,
            'gauss': ResizeFilter.Gauss
        }
        height, width = img.shape[:2]
        if self.width:
            height_k = height / width * self.size
            if width <= self.size:
                new_width = width * self.percent
                new_height = height * self.percent
            elif height < width and self.spread and self.spread_size < width:
                new_width = self.spread_size
                new_height = height / width * self.spread_size
            else:
                new_width = self.size
                new_height = height_k
        else:
            width_k = width / height * self.size
            if height <= self.size:
                new_width = width * self.percent
                new_height = height * self.percent
            else:
                new_width = width_k
                new_height = self.size
        img = resize(img.astype(np.float32), (int(new_width), int(new_height)), interpolation_map[self.interpolation],
                     gamma_correction=False)
        return img
