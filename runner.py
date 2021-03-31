import detector
import cv2
import numpy as np
import displayer
from pathlib import Path

in_dir = "imgs/"
out_dir = "imgs/out/"

class FattyImg:
    AREA_TYPES = ['fatty_cell', 'blood_vessel']

    def __init__(self, img_file):
        self.img_file = Path(img_file)
        self.img = cv2.imread(img_file)
        self.basic_structure = None
        self.clx = None

    def _return_basic_structure_propery(self, property):
        if self.basic_structure:
            return self.basic_structure[property]
        else:
            raise AttributeError("Basic structure not calculated or loaded")

    @property
    def shapes(self):
        return self._return_basic_structure_propery("shapes")

    @property
    def contours(self):
        return self._return_basic_structure_propery("contours")

    @property
    def params(self):
        return self._return_basic_structure_propery("params")

    def _calc_basic_structure(self):
        return detector.setup_img(self.img)

    def _assign_basic_structure(self, shapes, contours, params):
        self.basic_structure = {'shapes': shapes, 'contours': contours, 'params': params}

    def process_fully(self):
        shapes, contours, params, clx = detector.guess_shapes(cv2.imread(self.img_file))
        self._assign_basic_structure(shapes, contours, params)
        self.clx = clx

    def _get_basic_data_path(self):
        path = Path(self.img_file)
        return path.parent / path.stem

    def _get_basic_data_npz_path(self):
        return self._get_basic_data_path().with_suffix(".npz")

    def _does_basic_data_file_exist(self):
        return self._get_basic_data_npz_path().exists()

    def save_basic_data_to_file(self, overwrite_existing=True):
        if (not overwrite_existing) and self._does_basic_data_file_exist():
            raise ValueError(f"Basic file data npz already exists for {self.img_file}, and should not be overwritten.")
        else:
            np.savez_compressed(self._get_basic_data_path(), shapes=self.basic_structure['shapes'],
                            contours=self.basic_structure['contours'], params=self.basic_structure['params'])

    def load_basic_data_from_file(self):
        if self._does_basic_data_file_exist():
            self.basic_structure = np.load(self._get_basic_data_npz_path())
        else:
            raise FileNotFoundError(f"Could not find basic data for {self.img_file}")

    #by default, will always attempt to save the basic structure w/o overwritting.
    def find_basic_structure(self, save_basic_structure=True):
        if not self.basic_structure:
            try:
                self.load_basic_data_from_file()
            except FileNotFoundError:
                self._assign_basic_structure(*self._calc_basic_structure())
                self.save_basic_data_to_file(overwrite_existing=False)


    def get_img(self):
        return self.img

    def get_classified_img(self):
        return displayer.color_classified_shapes(np.copy(self.img), self.shapes, self.params, self.clx)

    def get_marked_img(self, marked_shapes):
        return displayer.color_classified_shapes(np.copy(self.img), self.shapes, self.params, marked_shapes)

    def get_pixel_count(self):
        num_of_white_pxls = detector.get_num_of_white_pxls(self.img)
        num_of_detected_pxls = np.sum(self.params[np.isin(self.params['id'], np.concatenate(self.clx))]['size'])
        return num_of_white_pxls, num_of_detected_pxls


def run_on_folder(folder_base):
    base_path = in_dir + folder_base + "/"
    p = Path(base_path)
    ratio_log_file = str(p) + "/ratios.txt"
    ratios = []
    for img_file in list(map(str, p.glob("**/*.*"))):
        print(f"Working on {img_file}....")
        file_parts = img_file.split(".")
        out_file = file_parts[0] + "_OUT." + file_parts[-1]
        fatty_img = FattyImg(img_file)
        fatty_img.process_fully()
        displayer.save_fig(fatty_img.get_classified_img(), out_file)
        num_of_white_pxls, num_of_detected_pxls = fatty_img.get_pixel_count()
        ratio = num_of_detected_pxls / num_of_white_pxls
        ratios.append((img_file, num_of_white_pxls, num_of_detected_pxls, ratio))
        print(f"Saved to {out_file}")
    with open(ratio_log_file, "w") as ratio_log:
        for img_file, num_of_white_pxls, num_of_detected, ratio in ratios:
            ratio_log.write(f"{img_file} - original {num_of_white_pxls}, detected {num_of_detected}, ratio {ratio}\n")
