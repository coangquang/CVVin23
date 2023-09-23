import os
import glob
def mask_image_list(dir = None):
    name_image = dir + '-Image'
    name_mask = dir + '-Mask'
    data_dir = os.path.join(os.getcwd(), dir + os.sep)
    img_dir = os.path.join(name_image + os.sep)
    mask_dir = os.path.join(name_mask + os.sep)

    img_ext, mask_ext = ".jpg", ".png"

    img_list = glob.glob(data_dir + img_dir + "*" + img_ext)
    mask_list = []

    for img_path in img_list:
        full_name = img_path.split(os.sep)[-1]

        name_ext = full_name.split(".")
        name_list = name_ext[0:-1]
        img_idx = name_list[0]

        for i in range(1, len(name_list)):
            img_idx = img_idx + "." + name_list[i]

        mask_list.append(data_dir + mask_dir + img_idx + mask_ext)

    return img_list, mask_list
