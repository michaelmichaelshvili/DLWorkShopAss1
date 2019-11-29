import os
import cv2
import sys


def get_minimum_image_sizes(images_path):
    minimum_axis_size = [sys.maxsize, sys.maxsize, sys.maxsize]
    for idx, image_name in enumerate(os.listdir(images_path)):
        print('idx:' + str(idx))
        raw_image = cv2.imread(os.path.join(images_path, image_name))
        for i in range(3):
            if raw_image.shape[i] < minimum_axis_size[i]:
                minimum_axis_size[i] = raw_image.shape[i]
    return minimum_axis_size


def resize_all_images(images_dir_path, output_dir_path, x_size, y_size):
    for idx, image_name in enumerate(os.listdir(images_dir_path)):
        print('idx:' + str(idx))
        raw_image = cv2.imread(os.path.join(images_dir_path, image_name))
        resize_img = cv2.resize(raw_image,(x_size,y_size))
        cv2.imwrite(os.path.join(output_dir_path, image_name), resize_img)








# print(get_minimum_image_sizes(r"dog-breed-identification\train"))  # [102, 97, 3]
# print(get_minimum_image_sizes(r"dog-breed-identification\stam"))  # [143, 190, 3]
resize_all_images(r"dog-breed-identification\test", r"dog-breed-identification\test_resize", 250, 250)


