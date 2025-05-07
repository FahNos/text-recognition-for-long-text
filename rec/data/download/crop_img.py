import os
import random
import cv2
import numpy as np
import copy

def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

def print_draw_crop_rec_res( img_crop_list, img_name, texts, img_crop_folder, crop_label):

    bbox_num = len(img_crop_list)
    for bno in range(bbox_num):
      crop_name=img_name+'_'+str(bno)+'.jpg'
      crop_name_w = f"{img_crop_folder}/{crop_name}"
      text = texts[bno].rstrip('\n')
      cv2.imwrite(crop_name_w, img_crop_list[bno])
      crop_label.write(f"{crop_name_w}\t{text}\n")

def get_txt_file_paths(folder_path):
    txt_file_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            txt_file_paths.append(file_path)
    return txt_file_paths

def process_image_file(image_file, dt_boxes, texts, crop_img_folder, crop_label):

    if not os.path.exists(image_file):
        print(f"Error: Image file not found: {image_file}")
        return False

    img = cv2.imread(image_file)
    if img is None:
        print(f"Error: Could not read image file: {image_file}")
        return False

    ori_im = img.copy()
    img_crop_list = []

    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = get_rotate_crop_image(ori_im, tmp_box)
        img_crop_list.append(img_crop)

    img_name = os.path.splitext(os.path.basename(image_file))[0]
    print_draw_crop_rec_res(img_crop_list, img_name, texts, crop_img_folder, crop_label)
    return True