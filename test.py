from keypoint import build_stacked_hourglass
from keypoint import keypoint_loss, focal_loss
from DataLoader import load_data_train, load_data_candidate, visualize_img_and_mask
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tqdm import tqdm
import os
import numpy as np
import json
import cv2
import matplotlib 
import matplotlib.pyplot as plt
import os
import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import backend as K

class Cfg:
    class Util:
        model_generation  = 2  # change this value if don't want old model overwritten
        img_dir           = "./12/"
        label_path        = "./12.json"
        candidates_dir    = "./12/"
        img_data_shape    = (0, 256,256, 3)
        mask_data_shape   = (0, 256, 256, 7)
        channel           = {"eye": 0, "mouth": 1, "backfin": 2, "chestfin": 3, "analfin": 4, "tail": 5, "backfin2": 6}
        keypoint_gaussian = True
        do_training       = True
        do_predicting     = False
    class Model:
        dim_output        = 7
        n_hourglass       = 3
        n_hourglass_layer = 3
        dim_feature       = 128
        jumpwire_mode     = 'concat'
        final_activation  = 'sigmoid'
    class Train:
        batch_size       = 8
        epochs           = 1
        validation_split = 0.1
        multi_GPU        = False
        lr               = 0.001
        decay            = 1e-6


def train(img_dir, label_path,
          model_save_path='model.h5', history_save_path='history.png'):
    dim_output        = Cfg.Model.dim_output
    n_hourglass       = Cfg.Model.n_hourglass
    n_hourglass_layer = Cfg.Model.n_hourglass_layer
    dim_feature       = Cfg.Model.dim_feature
    jumpwire_mode     = Cfg.Model.jumpwire_mode
    final_activation  = Cfg.Model.final_activation
    batch_size = Cfg.Train.batch_size
    epochs     = Cfg.Train.epochs
    multi_GPU  = Cfg.Train.multi_GPU
    lr         = Cfg.Train.lr
    decay      = Cfg.Train.decay
    img_data_shape  = Cfg.Util.img_data_shape
    mask_data_shape = Cfg.Util.mask_data_shape
    channel         = Cfg.Util.channel
    gaussian        = Cfg.Util.keypoint_gaussian

    # load data
    print('loading data...')
    imgs, img_files, img_orisizes, masks = load_data_train(img_dir=img_dir, label_path=label_path,
                                                           img_data_shape=img_data_shape, mask_data_shape=mask_data_shape,
                                                           keypoint_channel_hash=channel,
                                                           gaussian=gaussian)
    maskss = []
    for i in range(n_hourglass):
        maskss.append(masks)
        
    # build model
    print('building model...')
    model = build_stacked_hourglass(input_shape=img_data_shape[1:],
                                    dim_output=dim_output,
                                    n_hourglass=n_hourglass, n_hourglass_layer=n_hourglass_layer,
                                    dim_feature=dim_feature,
                                    jumpwire_mode=jumpwire_mode,
                                    final_activation=final_activation)
    if multi_GPU:
        model = multi_gpu_model(model, gpus=3)

    # compile model
    loss_weights = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]  # weakening intermediate supervision
    # loss_weights = [1, 1, 1, 1, 1, 1, 1]  # full intermediate supervision
    # loss_weights = [0, 0, 0, 0, 0, 0, 1]  # no intermediate supervision
    loss_weights = loss_weights[-n_hourglass:]
    loss_funcs = []
    for i in range(n_hourglass):
        loss_funcs.append(keypoint_loss())
    optimizer_adam = Adam(lr=lr, decay=decay)
    model.compile(optimizer=optimizer_adam,
                  loss=loss_funcs,
                  loss_weights=loss_weights)

    # training
    print('training model...')
    callback = ModelCheckpoint(filepath=model_save_path[:-3] + '_best.h5', monitor='val_loss', save_best_only=True)
    h = model.fit(imgs, maskss, validation_split=0.1,
                  batch_size=batch_size, epochs=epochs,
                  callbacks=[callback], shuffle=True)

    # save model
    print('saving model...')
    model.save(model_save_path)
    visualize_hist(h, save_path=history_save_path)

    # save model weights
    # usable if cannot load structured model due to keras version difference
    print('saving model weights...')
    model.save_weights(model_save_path[:-3] + ".weights.h5")
    model_best = load_model(model_save_path[:-3] + '_best.h5', custom_objects={'loss_func_keypoint': keypoint_loss()})
    model_best.save_weights(model_save_path[:-3] + '_best.weights.h5')


def predict(model_paths, res_save_dirs, json_res_save_paths, confidence_threshold=0.5):
    candidates_dir = Cfg.Util.candidates_dir
    img_data_shape  = Cfg.Util.img_data_shape
    mask_data_shape = Cfg.Util.mask_data_shape
    n_hourglass = Cfg.Model.n_hourglass
    keypoint_channel = Cfg.Util.channel

    print('loading image...')
    imgs, img_files, img_orisizes = load_data_candidate(img_dir=candidates_dir, img_data_shape=img_data_shape)

    predict_task = zip(model_paths, res_save_dirs, json_res_save_paths)
    for (model_path, res_save_dir, json_res_save_path) in predict_task:
        if not os.path.exists(model_path):
            continue
        if not os.path.exists(res_save_dir):
            os.makedirs(res_save_dir)
        json_res = {n: {} for n in range(n_hourglass)}
        print('loading model: ', os.path.split(model_path)[1], '...')
        model = load_model(model_path, custom_objects={'loss_func_keypoint': keypoint_loss()})  # without custom_objects, model won't recognize the loss defined by user
        print('predicting...')
        m = imgs.shape[0]
        for i in tqdm(range(m)):
            img = imgs[i:i + 1]
            img_file = img_files[i]
            orisize = img_orisizes[i]
            json_index = "{}{}".format(img_file, orisize[0]*orisize[1])
            ress = model.predict(img)
            for res, j in zip(ress, range(n_hourglass)):
                json_res[j][json_index] = {"filename": img_file,
                                           "size": orisize[0] * orisize[1],
                                           "regions": []}
                # mask = np.where(res > 0.5, 1, 0)
                mask = np.zeros((1, mask_data_shape[-3], mask_data_shape[-2], 0))
                for ch in range(res.shape[-1]):
                    keypoint_name = [k for k, v in keypoint_channel.items() if v == ch][0]
                    mask_single = res[:, :, :, ch:ch+1]
                    mask_single = np.where(mask_single == np.max(mask_single), mask_single, 0)
                    mask_single = np.where(mask_single > confidence_threshold, 1, 0)
                    mask = np.concatenate((mask, mask_single), axis=-1)
                    _, y, x, _ = np.where(mask_single == 1)
                    if len(x) > 0:
                        x, y = x[0] * (orisize[0] / mask_data_shape[-2]), y[0] * (orisize[1] / mask_data_shape[-3])
                        shape_attr = {"name": "point", "cx": x, "cy": y}
                        region_attr = {"keypoint": keypoint_name}
                        json_res[j][json_index]["regions"].append({"shape_attributes": shape_attr, "region_attributes": region_attr})
                res_save_path = res_save_dir + '/keypoint_predicted_' + str(j) + '_' + img_file
                visualize_img_and_mask(img, mask, orisize, save_path=res_save_path)
            for n in range(n_hourglass):
                with open(json_res_save_path[:-5]+".stage{}".format(n)+".json", "w") as f:
                    json.dump(json_res[n], f)


def visualize_hist(h, save_path=None):
    plt.figure()
    plt.plot(np.array(h.history['loss']))
    plt.plot(np.array(h.history['val_loss']))
    plt.title('LOSS')
    plt.ylabel('keypoint loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def main():
    model_generation = Cfg.Util.model_generation
    img_dir = Cfg.Util.img_dir
    label_path = Cfg.Util.label_path
    do_training   = Cfg.Util.do_training
    do_predicting = Cfg.Util.do_predicting

    model_save_path = 'keypoint for paper/models/model_' + str(model_generation) + '.h5'
    history_save_path = 'keypoint for paper/histories/history_' + str(model_generation) + '.png'
    if do_training:
        train(img_dir=img_dir, label_path=label_path,
              model_save_path=model_save_path, history_save_path=history_save_path)

    res_save_dir = 'keypoint for paper/results/predicted' + str(model_generation) + '/predicted'
    if do_predicting:
        predict(model_paths=[#model_save_path,
                             model_save_path[:-3] + '_best.h5',
                             model_save_path[:-3] + '_finetune.h5',
                             model_save_path[:-3] + '_finetune_best.h5'],
                res_save_dirs=[#res_save_dir,
                               res_save_dir + '_best',
                               res_save_dir + '_finetune',
                               res_save_dir + '_finetune_best'],
                json_res_save_paths=[#res_save_dir + "res.json",
                                     res_save_dir + "res_best.json",
                                     res_save_dir + "res_finetune.json",
                                     res_save_dir + "res_finetune_best.json",])


if __name__ == '__main__':
    main()
