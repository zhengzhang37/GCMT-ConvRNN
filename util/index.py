true_test_root = '/mnt/A/CIKM2017/CIKM_datasets/test/'
true_validation_root = '/mnt/A/CIKM2017/CIKM_datasets/validation/'
pred_root = '/mnt/A/meteorological/2500_ref_seq/'
import os
import sys
import cv2
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)
# import matplotlib.pyplot as plt
# from scipy.misc import imread
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided as ast
import math
import numpy as np
from scipy import stats
# from VarFlow.varflow.varflow import VarFlowFactory
# from skimage.measure import compare_ssim,compare_psnr,compare_mse



def generate_weight(true_imgs):
    import copy
    base_weight = np.zeros_like(true_imgs)
    w1 = copy.deepcopy(true_imgs)
    w1[w1>15] = 0
    w1[w1<=15] = 1
    base_weight += w1
    w2 = copy.deepcopy(true_imgs)
    w2[w2 <= 15] = 0
    w2[w2 > 25] = 0
    w2[w2 > 0] = 2
    base_weight += w2
    w3 = copy.deepcopy(true_imgs)
    w3[w3 <= 25] = 0
    w3[w3 > 40] = 0
    w3[w3 > 0] = 5
    base_weight += w3
    w4 = copy.deepcopy(true_imgs)
    w4[w4 < 40] = 0
    w4[w4 > 0] = 10
    base_weight += w4

    del w1
    del w2
    del w3

    return base_weight

def pixel_to_dBZ(data):
    dBZ = data.astype(np.float) * 95 / 255.0 - 10
    return dBZ

def eval_test(true_fold,pred_fold):
    bmses = []
    res = 0
    # valid_root_path = '/home/ices/PycharmProject/MultistageConvRNN/data/evaluate/valid_test.txt'
    # with open(valid_root_path) as f:
    #     sample_indexes = f.read().split('\n')[:-1]
    sample_indexes = list(range(1,4001,1))
    for index in sample_indexes:

        true_current_fold = true_fold+'sample_'+str(index)+'/'
        pre_current_fold = pred_fold+'sample_'+str(index)+'/'
        pred_imgs = []
        true_imgs = []
        for t in range(6, 16, 1):
            pred_path = pre_current_fold+'img_'+str(t)+'.png'
            true_path = true_current_fold+'img_'+str(t)+'.png'
            pred_img = cv2.imread(pred_path,0)
            true_img = cv2.imread(true_path,0)
            pred_img = pred_img.astype(np.float32)
            true_img = true_img.astype(np.float32)
            pred_imgs.append(pred_img)
            true_imgs.append(true_img)
        pred_imgs = np.array(pred_imgs)
        true_imgs = np.array(true_imgs)


        pred_imgs = pixel_to_dBZ(pred_imgs)
        true_imgs = pixel_to_dBZ(true_imgs)
        weight = generate_weight(true_imgs)
        sample_res = np.mean(weight*np.square(pred_imgs - true_imgs))

        bmses.append(sample_res)
        res = res+sample_res
    res = res/len(sample_indexes)
    np.save('multi_trajgru.npy',bmses)
    return res


if __name__ == '__main__':
    # test_model_list = [
    #     # "TrajGRU_test",
    #     "wrap_TrajGRU_test",
    # ]
    # test_model_bmse = {}
    # for model in test_model_list:
    #     mae = eval_test(true_test_root, pred_root + model + '/')
    #     test_model_bmse[model] = mae
    #     print('b-mse model is:', model)
    #     print(test_model_bmse[model])
    trajgru_bmses = np.load('/mnt/B/zhangz/multi_task_CIKM/util/trajgru.npy')
    multi_trajgru_bmses = np.load('/mnt/B/zhangz/multi_task_CIKM/util/multi_trajgru.npy')
    # print(multi_trajgru_bmses[2360], trajgru_bmses[2360])
    # print(np.argmin(trajgru_bmses), np.min(trajgru_bmses))
    ind = np.argpartition(multi_trajgru_bmses - trajgru_bmses, 10)[:30]
    # print(np.argmin(multi_trajgru_bmses - trajgru_bmses), np.min(multi_trajgru_bmses - trajgru_bmses))
    print(ind)