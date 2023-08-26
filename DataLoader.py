import numpy as np
import cv2
import os
import json


def load_data_train(img_dir, label_path,
                    keypoint_channel_hash,
                    img_data_shape=(0, 256, 256, 3), mask_data_shape=(0, 256, 256, 3),
                    gaussian=True,  # If set True，add a Gaussian mask to the keypoint, for better orientating of the gradient. Set this False if the model is not predicting keypoints.
                    visualize=False):
    imgs = np.zeros(img_data_shape, dtype=np.float)
    img_files = []
    img_orisizes = []  # original size noted in cv2 format
    masks = np.zeros(mask_data_shape, dtype=np.float)

    keypointss = load_keypoints_from_label(label_path)
    l = 0
    for keypoints in keypointss:
        img_path = img_dir + '/' + keypoints['filename']
        if os.path.exists(img_path):
            # operation with img
            img_files.append(img_path)
            img = cv2.imread(img_path)
            img_orisize = (img.shape[1], img.shape[0])
            img_orisizes.append(img_orisize)                                # should remember each img's original size, for resizing them back
            img = cv2.resize(img, (img_data_shape[2], img_data_shape[1]))   # in cv2, columns first, rows second
            img = img.astype(np.float)                                      # convert img from int to float
            img = img/255 - 0.5                                             # make pixel value between -0.5 ~ 0.5
            img = np.expand_dims(img, axis=0)                               # fit the shape of training set
            imgs = np.concatenate((imgs, img), axis=0)                      # put img into training set

            # operation with mask
            mask = np.zeros(mask_data_shape[1:], dtype=np.float)
            for point_class in keypoints['points']:
                i = keypoint_channel_hash[point_class]
                keypoint = keypoints['points'][point_class]
                where_ori_row = keypoint[1]
                where_ori_col = keypoint[0]
                where_row = int(where_ori_row * mask_data_shape[1] / img_orisize[1])
                where_col = int(where_ori_col * mask_data_shape[2] / img_orisize[0])
                mask[where_row, where_col, i] = 1
                pass
            mask = np.expand_dims(mask, axis=0)
            if gaussian:
                mask = gen_keypoint_gaussian_mask(mask)
            masks = np.concatenate((masks, mask), axis=0)

            if visualize:
                visualize_img_and_mask(img, mask, img_orisize)
        l = l +1
        print('\r'+str(l),end='')
    return imgs, img_files, img_orisizes, masks


def load_data_candidate(img_dir='data/candidates', img_data_shape=(0, 64, 128, 3)):
    imgs = np.zeros(img_data_shape, dtype=np.float)
    img_orisizes = []
    img_files = os.listdir(img_dir)
    for img_file in img_files:
        img_path = img_dir + '/' + img_file
        img = cv2.imread(img_path)
        img_orisize = (img.shape[1], img.shape[0])
        img_orisizes.append(img_orisize)                               # should remember each img's original size, for resizing them back
        img = cv2.resize(img, (img_data_shape[2], img_data_shape[1]))  # in cv2, columns first, rows second
        img = img.astype(np.float)                                     # convert img from int to float
        img = img / 255 - 0.5                                          # make pixel value between -0.5 ~ 0.5
        img = np.expand_dims(img, axis=0)                              # fit the shape of training set
        imgs = np.concatenate((imgs, img), axis=0)                     # put img into training set

    return imgs, img_files, img_orisizes


def load_keypoints_from_label(label_path):
    f = open(label_path, encoding='utf-8')
    labels = json.load(f)
    keypointss = []
    l = 0
    for file in labels:
        keypoints = dict()
        keypoints['filename'] = labels[file]['filename']
        keypoints['points'] = dict()
        for point in labels[file]['regions']:
            category = point['region_attributes']['keypoint']
            coord = (int(point['shape_attributes']['cx']), int(point['shape_attributes']['cy']))
            keypoints['points'][category] = coord
        keypointss.append(keypoints)
    return keypointss


def visualize_img_and_mask(img, mask, orisize, save_path=None):
    # Resize img back to its original size
    img = np.squeeze(img)
    img = (img + 0.5)*255
    img = img.astype(np.uint8)
    img = cv2.resize(img, orisize)

    # Draw keypoints by mask
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 126, 126)]  # Add more colors if number of keypoints exceeds 7. Or use the same color.
    mask = np.squeeze(mask)
    for i in range(mask.shape[-1]):
        if i >= len(colors):
            color = (255, 255, 255)
        else:
            color = colors[i]
        mask_single = np.squeeze(mask[:, :, i])
        where_row, where_col = np.where(mask_single == 1)
        for row, col in zip(where_row, where_col):
            where_row_ori = int(row * orisize[1] / mask_single.shape[0])
            where_col_ori = int(col * orisize[0] / mask_single.shape[1])
            cv2.circle(img, center=(where_col_ori, where_row_ori), radius=2, color=color, thickness=2)
    if save_path is None:
        cv2.imshow('img with keypoints', img)
        cv2.waitKey(0)
    else:
        cv2.imwrite(save_path, img)


def gen_keypoint_gaussian_mask(mask, sigma=3):
    def cal_dist(point1, point2):
        point1 = np.array(point1)
        point2 = np.array(point2)
        diff = point1 - point2
        dist = np.matmul(diff, np.transpose(diff)) ** 0.5
        return dist

    def gaussian_kernel_func(point1, point2):
        dist = cal_dist(point1, point2)
        res = np.exp(- dist ** 2 / (2 * sigma ** 2))
        return res

    mask_gaussian = np.zeros_like(mask)  # (1, h, w, ch)
    for ch in range(mask.shape[-1]):
        mask_single = np.squeeze(mask[0, :, :, ch])
        where_row, where_col = np.where(mask_single == 1)
        for row, col in zip(where_row, where_col):
            for i in range(mask_single.shape[0]):
                for j in range(mask_single.shape[1]):
                    mask_gaussian[0, i, j, ch] = gaussian_kernel_func((i, j), (row, col))

    return mask_gaussian


if __name__ == '__main__':
    # Keypoint label is from labelling tool "via".
    load_data_train(img_dir="./roi_ss/",
                    label_path="11.json",
                    keypoint_channel_hash={"eye": 0, "mouth": 1, "backfin": 2, "chestfin": 3, "analfin": 4, "tail": 5, "backfin2": 6},
                    mask_data_shape=(0, 256, 256, 7),
                    visualize=False)
