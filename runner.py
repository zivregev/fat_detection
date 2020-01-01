import detector
import cv2
import numpy as np
import displayer
from pathlib import Path

in_dir = "imgs/"
out_dir = "imgs/out/"

def run_on_folder(folder_base):
    base_path = in_dir + folder_base + "/"
    p = Path(base_path)
    ratio_log_file = str(p) + "/ratios.txt"
    ratios = []
    for img_file in list(map(str, p.glob("**/*.*"))):
        print(f"Working on {img_file}....")
        file_parts = img_file.split(".")
        out_file = file_parts[0] + "_OUT." + file_parts[-1]
        img = cv2.imread(img_file)
        shapes, contours, params, clx = detector.guess_shapes(img)
        displayer.save_fig(displayer.color_classified_shapes(np.copy(img), shapes, params, clx), out_file)
        num_of_white_pxls = detector.get_num_of_white_pxls(img)
        num_of_detected_pxls = np.sum(params[np.isin(params['id'], np.concatenate(clx))]['size'])
        ratio = num_of_detected_pxls / num_of_white_pxls
        ratios.append((img_file, num_of_white_pxls, num_of_detected_pxls, ratio))
        print(f"Saved to {out_file}")
    with open(ratio_log_file, "w") as ratio_log:
        for img_file, num_of_white_pxls, num_of_detected, ratio in ratios:
            ratio_log.write(f"{img_file} - original {num_of_white_pxls}, detected {num_of_detected}, ratio {ratio}\n")
