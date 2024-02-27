from src.sharp.sharp import Sharp
from src.halftone.halftone import Halftone
from src.resize.resize import Resize
import cv2
import numpy as np
import argparse
import os
import json
from tqdm.contrib.concurrent import process_map


class Start:
    def __init__(self):
        self.in_folder = ""
        self.out_folder = str
        self.sharp = Sharp

    def __arg_parse(self) -> None:
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('-i', '--input', type=str,
                            help='Input_folder')
        parser.add_argument('-o', '--output', type=str,
                            help='Output_folder')

        args = parser.parse_args()
        in_folder = args.input
        out_folder = args.output
        if not in_folder:
            in_folder = "INPUT"
        if not out_folder:
            out_folder = "OUTPUT"
        self.in_folder = in_folder
        self.out_folder = out_folder
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        if not os.path.exists(in_folder):
            os.makedirs(in_folder)
            raise print("no in folder")

    def __json_parse(self) -> None:
        with open("config.json", "r") as f:
            json_config = json.load(f)
        if list(json_config.keys()) != ['low_input', 'high_input', 'gamma', 'diapason_white', 'cenny', 'dot_size',
                                        'size', 'interpolation', 'width', 'percent', 'spread', 'spread_size']:
            raise print('Not correct config')
        diapason_white = json_config["diapason_white"]
        low_input = json_config["low_input"]
        high_input = json_config["high_input"]
        gamma = json_config["gamma"]
        cenny = json_config["cenny"]
        dot_size = json_config["dot_size"]
        size = json_config["size"]
        interpolation = json_config["interpolation"]
        width = json_config["width"]
        percent = json_config["percent"]
        spread = json_config["spread"]
        spread_size = json_config["spread_size"]
        try:

            self.sharp = Sharp(diapason_white, low_input, high_input, gamma, cenny)

            self.halftone = Halftone(dot_size)

            self.resize = Resize(size, interpolation, width, percent, spread, spread_size)

        except RuntimeError as e:
            raise print(f"incorrect data type {e}")
        pass

    def process_img(self, img_name):
        try:
            folder = f"{self.in_folder}/{img_name}"
            basename = ".".join(img_name.split(".")[:-1])
            img = cv2.imread(folder)

            if img is None:
                return print(f"{img_name}, not supported")
            array = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
            array = self.sharp.run(array)
            array = self.halftone.run(array)
            array = self.resize.run(array)
            cv2.imwrite(f'{self.out_folder}/{basename}.png', array * 255)
        except RuntimeError as e:
            print(e)

    def start_process(self) -> None:
        self.__arg_parse()
        self.__json_parse()
        list_files = [
            file
            for file in os.listdir(self.in_folder)
            if os.path.isfile(os.path.join(self.in_folder, file))
        ]
        process_map(self.process_img, list_files)


if __name__ == "__main__":
    s = Start()
    s.start_process()
