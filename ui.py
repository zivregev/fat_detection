from tkinter import Tk, filedialog
from runner import FattyImg
import cv2
import os

class SingleImageEditor:

    TAGGED_IMG_SUFFIX = "_tagged"

    def __init__(self, img_file):
        self.img_name = img_file
        self.window_name = str(self.img_name)
        self.fatty_img = FattyImg(img_file)
        self.confirmed_shapes = {}
        self.confirmed_pxls = []
        self.prepare_img()

    def prepare_img(self):
        self.fatty_img.find_basic_structure()

    def edit(self):
        cv2.namedWindow(winname=self.window_name)
        self.mark_area_types()


    def handle_marked_pxl(self, pxl):
        shape_id = self.fatty_img.shapes[pxl]
        updated = False
        if shape_id > 0:
            try:
                area_type = self.confirmed_shapes[shape_id] + 1
            except KeyError:
                area_type = 0
            try:
                if area_type >= len(FattyImg.AREA_TYPES):
                    del self.confirmed_shapes[shape_id]
                else:
                    self.confirmed_shapes[shape_id] = area_type
                    self.confirmed_pxls.append(pxl)
                updated = True
            except KeyError:
                updated = False
        return updated

    def _get_updated_img(self):
        if len(self.confirmed_shapes) > 0:
            clx_by_mark = [[] for type in FattyImg.AREA_TYPES]
            for k, v in self.confirmed_shapes.items():
                clx_by_mark[v].append(k)
            updated_img = self.fatty_img.get_marked_img(clx_by_mark)
        else:
            updated_img = self.fatty_img.get_img()
        return updated_img

    def mark_area_types(self):
        for i in range(len(FattyImg.AREA_TYPES)):
            print(f"Double click on any area {i} times to mark it as {FattyImg.AREA_TYPES[i]}")
        print(f"Double click on any area {len(FattyImg.AREA_TYPES)} times to unmark it.")
        clicked_pxls = []
        updated = True
        while True:
            if clicked_pxls:
                pxl = clicked_pxls.pop()
                updated = self.handle_marked_pxl(pxl)
            if updated:
                updated = False
                updated_img = self._get_updated_img()
                cv2.setMouseCallback(self.window_name, SingleImageEditor.on_area_click, param=clicked_pxls)
                cv2.imshow(self.window_name, updated_img)
            key = cv2.waitKey(1)
            if key != -1:
                break
        save_img = get_single_char_input(f"Save tagged image y/n? \n", ['y', 'n'])
        if save_img:
            self.save_tagged_img(self._get_updated_img(), self.img_name)
        cv2.destroyWindow(self.window_name)

    def on_area_click(event, x, y, flags, clicked_pxls):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            clicked_pxls.append((y, x))

    def save_tagged_img(self, img, untagged_file_name):
        root, ext = os.path.splitext(untagged_file_name)
        root += SingleImageEditor.TAGGED_IMG_SUFFIX
        cv2.imwrite(root + ext, img)

def get_image_list_from_user():
    root = Tk()
    root.withdraw()
    img_list = filedialog.askopenfilenames(parent=root, title="Choose files to process")
    root.quit()
    return img_list

def preprocess_all_imgs(imgs_to_process):
    for img_file in imgs_to_process:
        print(f"Preprocessing {img_file}...")
        fatty_img = FattyImg(img_file)
        fatty_img.find_basic_structure()

def run_in_single_mode(imgs_to_process):
    for img_file in imgs_to_process:
        print(f"Working on {img_file}....")
        editor = SingleImageEditor(img_file)
        editor.edit()

def get_single_char_input(msg, options):
    while True:
        user_input = input(msg)
        if len(user_input) != 1 or user_input[0] not in options:
            pass
        else:
            return user_input[0]


if __name__ == "__main__":
    op_mode = get_single_char_input("Enter [p] to preprocess all images, or [s] to work in single mode. \n", ['p', 's'])
    imgs_to_process = get_image_list_from_user()
    continue_to_classify = (op_mode == 's')
    if op_mode == 'p':
        continue_after = get_single_char_input("Would you like to continue with classification after preprocessing (Or quit and come back later)? y/n: \n", ['y', 'n'])
        preprocess_all_imgs(imgs_to_process)
        if continue_after == 'y':
            continue_to_classify = True
    if continue_to_classify:
        run_in_single_mode(imgs_to_process)