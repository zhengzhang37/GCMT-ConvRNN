

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
from skimage.measure import compare_ssim,compare_psnr,compare_mse
import pandas as pd


from math import sqrt
# def _tf_fspecial_gauss(size, sigma):
#     """Function to mimic the 'fspecial' gaussian MATLAB function
#     """
#     x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
#
#     x_data = np.expand_dims(x_data, axis=-1)
#     x_data = np.expand_dims(x_data, axis=-1)
#
#     y_data = np.expand_dims(y_data, axis=-1)
#     y_data = np.expand_dims(y_data, axis=-1)
#
#     x = tf.constant(x_data, dtype=tf.float32)
#     y = tf.constant(y_data, dtype=tf.float32)
#
#     g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
#     return g / tf.reduce_sum(g)
#
#
# def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
#     window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
#     K1 = 0.01
#     K2 = 0.03
#     K3 = 0.15
#     L = 1  # depth of image (255 in case the image has a differnt scale)
#     C1 = (K1*L)**2
#     C2 = (K2*L)**2
#     C3 = (K3*L)**2
#     mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='SAME')
#     mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='SAME')
#
#     mu1_sq = mu1*mu1
#     mu2_sq = mu2*mu2
#     mu1_mu2 = mu1*mu2
#     sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='SAME') - mu1_sq
#     sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='SAME') - mu2_sq
#     sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='SAME') - mu1_mu2
#     if cs_map:
#         value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
#                     (sigma1_sq + sigma2_sq + C2)),
#                 (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
#     else:
#         value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
#                     (sigma1_sq + sigma2_sq + C2))
#
#     if mean_metric:
#         value = tf.reduce_mean(value)
#     return value
#
#
# def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
#     weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
#     mssim = []
#     mcs = []
#     for l in range(level):
#         ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
#         mssim.append(tf.reduce_mean(ssim_map))
#         mcs.append(tf.reduce_mean(cs_map))
#         filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
#         filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
#         img1 = filtered_im1
#         img2 = filtered_im2
#
#     # list to tensor of dim D+1
#     mssim = tf.stack(mssim, axis=0)
#     mcs = tf.stack(mcs, axis=0)
#
#     value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
#                             (mssim[level-1]**weight[level-1]))
#
#     if mean_metric:
#         value = tf.reduce_mean(value)
#     return value
# def image_to_4d(image):
#     image = tf.expand_dims(image, 0)
#     image = tf.expand_dims(image, -1)
#     return image
#
# BATCH_SIZE = 1
# CHANNELS = 1
# image1 = tf.placeholder(tf.float32, shape=[101, 101])
# image2 = tf.placeholder(tf.float32, shape=[101, 101])
# image4d_1 = image_to_4d(image1)
# image4d_2 = image_to_4d(image2)
# ssim_index = tf_ssim(image4d_1, image4d_2)
# msssim_index = tf_ms_ssim(image4d_1, image4d_2)
#
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
#
# def TF_SSIM(img1,img2):
#     tf_ssim_ = sess.run(ssim_index,
#                             feed_dict={image1: img1, image2: img2})
#
#     tf_msssim_ = sess.run(msssim_index,
#                             feed_dict={image1: img1, image2: img2})
#
#     return tf_ssim_,tf_msssim_
#
#     # print('tf_ssim_none', tf_ssim)
#     # print('tf_msssim_none', tf_msssim)
#







def PCC(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    p = np.corrcoef(y_true, y_pred)[1,0]
    if math.isnan(p):
        return -1
    return p

    #
    # ux = np.mean(y_true)
    # uy = np.mean(y_pred)
    # var_x = np.var(y_true)
    # var_y = np.var(y_pred)
    # if var_x ==0 or var_y == 0:
    #     print('error',str(var_y),str(var_x))
    #     return -1
    # std_x = np.sqrt(var_x)
    # std_y = np.sqrt(var_y)
    # var_xy = np.sum(np.dot((y_true - ux), (y_pred - uy))) / (101 * 101 - 1)
    # return var_xy / (std_x*std_y)

def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (int(A.shape[0]/ block[0]), int(A.shape[1]/ block[1]))+ block
    strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)

def SSIM(y_true, y_pred, C1=0.01**2, C2=0.03**2, C3=0.02**2):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    ux = np.mean(y_true)
    uy = np.mean(y_pred)

    var_x = np.var(y_true)
    var_y = np.var(y_pred)
    std_x = np.sqrt(var_x)
    std_y = np.sqrt(var_y)
    # var_xy = np.mean(np.cov(y_true,y_pred))[0,1]
    var_xy = np.sum(np.dot((y_true-ux),(y_pred-uy)))/(101*101-1)

    l = ((2*ux*uy+C1)/(ux*ux+uy*uy+C1))
    c = ((2*std_x*std_y+C1)/(var_x+var_y+C2))
    s = ((var_xy+C3)/(std_x*std_y+C3))

    return l*c*s


def MSE(pre,real):
    return np.mean(np.square(pre-real))

def MAE(pre,real):
    return np.sum(np.absolute(pre - real))/len(pre.reshape(-1))

def normalization(data):
    return data.astype(np.float32)/255.0

def eval_validation(true_fold,pred_fold,eval_type):
    res = 0
    for i in range(1,2000,1):
        # print('complete ',str(i*100.0/2000),'%')
        true_current_fold = true_fold + 'sample_' + str(i) + '/'
        pre_current_fold = pred_fold + 'sample_' + str(i) + '/'
        sample_res = 0
        pred_imgs =[]
        true_imgs =[]
        for t in range(6,16,1):
            pred_path = pre_current_fold+'img_'+str(t)+'.png'
            true_path = true_current_fold+'img_'+str(t)+'.png'
            pred_img = cv2.imread(pred_path,0)
            true_img = cv2.imread(true_path,0)
            pred_imgs.append(pred_img)
            true_imgs.append(true_img)
        pred_imgs = np.array(pred_imgs)
        ture_imgs = np.array(ture_imgs)

        if eval_type == 'mse':
            current_res = np.mean(np.square(pred_imgs-true_imgs))
        elif eval_type == 'mae':
            current_res = np.mean(np.abs(pred_imgs-true_imgs))


    res = res / 2000
    return res

def pixel_to_dBZ(data):
    dBZ = data.astype(np.float) * 95 / 255.0 - 10
    return dBZ

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

def eval_test(true_fold,pred_fold,eval_type):
    res = 0
    valid_root_path = '/home/ices/home/ices/zz/multi_task_CIKM/util/valid_test.txt'
    with open(valid_root_path) as f:
        sample_indexes = f.read().split('\n')[:-1]
    # sample_indexes = list(range(1,4001,1))
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

        # pred_imgs = pred_imgs.astype(np.float)
        # true_imgs = true_imgs.astype(np.float)
        if eval_type == 'mse':
            pred_imgs = pixel_to_dBZ(pred_imgs)
            true_imgs = pixel_to_dBZ(true_imgs)
            # sample_res = np.square(pred_imgs - true_imgs).mean()
            sample_res = np.mean(np.square(pred_imgs - true_imgs))
        elif eval_type == 'b-mse':
            pred_imgs = pixel_to_dBZ(pred_imgs)
            true_imgs = pixel_to_dBZ(true_imgs)
            weight = generate_weight(true_imgs)
            sample_res = np.mean(weight*np.square(pred_imgs - true_imgs))
        elif eval_type == 'mae':
            pred_imgs = pixel_to_dBZ(pred_imgs)
            true_imgs = pixel_to_dBZ(true_imgs)
            # sample_res = np.abs(pred_imgs - true_imgs).mean()
            sample_res = np.mean(np.abs(pred_imgs - true_imgs))
        elif eval_type == 'b-mae':
            pred_imgs = pixel_to_dBZ(pred_imgs)
            true_imgs = pixel_to_dBZ(true_imgs)
            weight = generate_weight(true_imgs)
            sample_res = np.mean(weight*np.abs(pred_imgs - true_imgs))


        elif eval_type == 'ssim':
            pred_imgs = pred_imgs/255.0
            true_imgs =  true_imgs/255.0
            sample_res = 0
            for t in range(10):
                ssim = compare_ssim(pred_imgs[t],true_imgs[t])
                sample_res = sample_res+ssim
            sample_res = sample_res/10.0
        elif eval_type == 'psnr':
            sample_res = 0
            pred_imgs = pred_imgs.astype(np.uint8)
            true_imgs = true_imgs.astype(np.uint8)
            for t in range(10):
                psnr = compare_psnr(pred_imgs[t],true_imgs[t],data_range=255)
                sample_res = sample_res+psnr
            sample_res = sample_res/10.0

        res = res+sample_res
    res = res/len(sample_indexes)
    return res


def sequence_mse(true_fold,pred_fold,eval_type='mse'):
    res = [0 for _ in range(10)]
    # valid_root_path = '/home/ices/PycharmProject/MultistageConvRNN/data/evaluate/valid_test.txt'
    # with open(valid_root_path) as f:
    #     sample_indexes = f.read().split('\n')[:-1]
    sample_indexes = list(range(1,4001,1))

    for i in sample_indexes:
        # print('complete ',str(i*100.0/4000),' %')
        true_current_fold = true_fold + 'sample_' + str(i) + '/'
        pre_current_fold = pred_fold + 'sample_' + str(i) + '/'
        sample_res = []
        skip = 0
        for t in range(6, 16, 1):
            pred_path = pre_current_fold + 'img_' + str(t) + '.png'
            true_path = true_current_fold + 'img_' + str(t) + '.png'
            pre_img = imread(pred_path)
            true_img = imread(true_path)
            # pre_img = pixel_to_dBZ(pre_img)
            # true_img = pixel_to_dBZ(true_img)
            if eval_type == 'mse':
                current_res = np.square(pre_img - true_img).mean()
            elif eval_type == 'mae':
                current_res = np.abs(pre_img - true_img).mean()

            # elif eval_type == 'ssim':
            #     current_res = TF_SSIM(normalization(pre_img), normalization(true_img))[0]
            # elif eval_type == 'ms-ssim':
            #     current_res = TF_SSIM(normalization(pre_img), normalization(true_img))[1]
            #     if math.isnan(current_res):
            #         skip = skip + 1

            sample_res.append(current_res)

        for i in range(len(res)):
            res[i] = res[i]+sample_res[i]

    for i in range(len(res)):
        res[i] = res[i] / len(sample_indexes)
    return res

#
# def plot(datas,names):
#
#     x = []
#     for i in range(1, 11, 1):
#         x.append(i*6)
#     plt.figure(figsize=(7.5,5))
#     model_names = [
#         "ConvLSTM",
#         "ST-LSTM",
#         "TrajLSTM",
#         "ST-TrajLSTM",
#         # "DFN",
#         # "TrajGRU",
#         # "PredRNN",
#         # "PredRNN++",
#         # "MIM",
#         # "E3D_LSTM",
#         "PFST-LSTM"]
#     for idx,name in enumerate(names):
#         plt.plot(x,datas[name])
#         names[idx] = model_names[idx]+':'+str(np.mean(np.array(datas[name])))[:5]
#
#
#     plt.grid()
#     plt.legend(names)
#     plt.xticks(x)
#     plt.xlabel('Leadtime (Minutes)')
#     plt.ylabel('Mean Square Error (MSE)')
#     plt.savefig('evalute.png')
#     plt.show()
#

def record_info_extract(path):
    with open(path,"r") as f:
        data = f.read().split('\n')[:-1]
    HSS = []
    CSI = []
    POD = []
    FAR = []

    for t in range(len(data)):
        cur_ele = data[t].split('\t')[1:-1]
        HSS.append(cur_ele[0].split(','))
        CSI.append(cur_ele[1].split(','))
        POD.append(cur_ele[2].split(','))
        FAR.append(cur_ele[3].split(','))

    HSS = np.array(HSS)
    CSI = np.array(CSI)
    POD = np.array(POD)
    FAR = np.array(FAR)

    return HSS,CSI,POD,FAR

def record_sample_extract(path):

    with open(path, "r") as f:
        data = f.read().split('\n')[:-1]
    a = []
    b = []
    c = []
    d = []
    for t in range(len(data)):
        cur_ele = data[t].split('\t')[:-1]
        a.append(cur_ele[0].split(','))
        b.append(cur_ele[1].split(','))
        c.append(cur_ele[2].split(','))
        d.append(cur_ele[3].split(','))

    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)
    c = np.array(c).astype(np.float32)
    d = np.array(d).astype(np.float32)

    a_list = a[:,-1]
    b_list = b[:,-1]
    c_list = c[:,-1]
    d_list = d[:,-1]
    data_check(a_list)
    data_check(b_list)
    data_check(c_list)
    data_check(d_list)
    # print("the shape of hss_list is:",str(hss_list.shape))
    # print("the shape of csi_list is:",str(csi_list.shape))
    return a_list, b_list, c_list, d_list
#
# def plot_evaluate(res,res_name):
#     x = []
#     for i in range(1, 11, 1):
#         x.append(i * 6)
#     threasholds = ['5','20','40','avg']
#     model_names = [
#         # "ConvLSTM",
#         # "ST-LSTM",
#         # "TrajLSTM",
#         # "ST-TrajLSTM",
#         "ConvLSTM",
#         "ConvGRU",
#         "CDNA",
#         "DFN",
#         "TrajGRU",
#         "PredRNN",
#         "PredRNN++",
#         "MIM",
#         "E3D_LSTM",
#         "PFST-LSTM"
#     ]
#     for threashold_idx in range(res.shape[-1]):
#         cur_threashold = res[:,:,threashold_idx]
#         plt.figure(figsize=(7.5,5))
#         names = []
#         for method_idx in range(len(cur_threashold)):
#             cur_record = cur_threashold[method_idx]
#             plt.plot(x, cur_record)
#             cur_name = model_names[method_idx]
#             names.append(cur_name)
#
#         plt.grid()
#         plt.title('The '+res_name[-4:-1]+' as threshold = '+str(threasholds[threashold_idx]))
#         plt.legend(names)
#         plt.xticks(x)
#         plt.xlabel('Leadtime (Minutes)')
#         plt.ylabel(res_name)
#         plt.savefig(res_name[-4:-1]+'_'+str(threasholds[threashold_idx])+'.png')
#
def plot_res(model_list):

    record_root_path = '/home/ices/Downloads/'
    HSSs,CSIs,PODs,FARs = [],[],[],[]
    for model_name in model_list:
        cur_record_path = os.path.join(record_root_path, model_name + '.txt')
        HSS, CSI, POD, FAR = record_info_extract(cur_record_path)
        HSSs.append(HSS)
        CSIs.append(CSI)
        PODs.append(POD)
        FARs.append(FAR)
    HSSs = np.array(HSSs).astype(np.float32)
    CSIs = np.array(CSIs).astype(np.float32)
    # plot_evaluate(HSSs, 'Heidk Skill Score (HSS)')
    # plot_evaluate(CSIs, 'Critical Success Index (CSI)')


def data_check(data):
    for index,d in enumerate(data):
        if np.isnan(d):
            print('have None',str(index))


def t_test(data):
    print(data.shape)
    p_matrix = np.zeros((data.shape[0],data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            data1 = data[i]
            data2 = data[j]
            (statistic, pvalue) = stats.levene(data1,data2)
            print('p value is:',str(pvalue))
            p_matrix[i,j]=pvalue

    return p_matrix

def all_t_test(model_list):
    record_root_path = '/home/ices/Public/'
    As, Bs, Cs, Ds = [], [], [], []
    for model_name in model_list:
        cur_record_path = os.path.join(record_root_path, model_name + '.txt')
        a_list, b_list, c_list, d_list = record_sample_extract(cur_record_path)
        As.append(a_list)
        Bs.append(b_list)
        Cs.append(c_list)
        Ds.append(d_list)
    As = np.array(As).astype(np.float32)
    Bs = np.array(Bs).astype(np.float32)
    Cs = np.array(Cs).astype(np.float32)
    Ds = np.array(Ds).astype(np.float32)

    As_t_test = t_test(As)
    Bs_t_test = t_test(Bs)
    Cs_t_test = t_test(Cs)
    Ds_t_test = t_test(Ds)
    print(As_t_test)
    print(Bs_t_test)
    print(Cs_t_test)
    print(Ds_t_test)

    As_t_test[As_t_test > 0.05] = 1
    As_t_test[As_t_test <= 0.05] = 0
    print(As_t_test)
    # print(HSSs.shape,CSIs.shape)


def flow_test(test_model_list,s_index):
    test_root = '/mnt/A/meteorological/2500_ref_seq/'
    sample_index = 'sample_'+str(s_index)
    imgs = []
    real_cur_fold = os.path.join(true_test_root, sample_index)
    cur_real_imgs = []
    for t in range(6, 16, 1):
        cur_path = os.path.join(real_cur_fold, 'img_' + str(t) + '.png')
        img = imread(cur_path)
        cur_real_imgs.append(img)

    cur_real_imgs = np.array(cur_real_imgs)


    for model in test_model_list:
        cur_fold = os.path.join(test_root,model,sample_index)
        cur_imgs = []
        for t in range(6,16,1):
            cur_path = os.path.join(cur_fold,'img_'+str(t)+'.png')
            img = imread(cur_path)
            cur_imgs.append(img)
        cur_imgs = np.array(cur_imgs)
        imgs.append(cur_imgs)
    imgs.append(cur_real_imgs)
    imgs = np.array(imgs)




    varflow_factory = VarFlowFactory(max_level=4, start_level=0, n1=2, n2=2, rho=2.8, alpha=1400, sigma=1.5)
    flows = []
    means_flows = []
    for model_ind in range(len(imgs)):
        cur_model_imgs = imgs[model_ind]
        cur_flows = []
        for t in range(10-1):
            m1 = cur_model_imgs[t]
            m2 = cur_model_imgs[t+1]
            flow = varflow_factory.calc_flow(I1=m1,I2=m2)

            df1 = pd.DataFrame(flow[0])
            fd1 = df1.fillna(0).values
            df2 = pd.DataFrame(flow[1])
            df2 = df2.fillna(0).values
            fd = np.stack([fd1,df2],0)
            cur_flows.append(np.abs(fd))

        cur_flows = np.array(cur_flows)
        means_flows.append(np.mean(cur_flows))
        flows.append(cur_flows)

    flows = np.array(flows)

    return np.array(means_flows)



def calculate_ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
if __name__ == '__main__':
    # import os
    # import cv2
    # pred_img = cv2.imread('/mnt/A/meteorological/2500_ref_seq/CIKM_rst_lstm_test/sample_339/img_6.png',0).astype(np.float64)
    # gt_img = cv2.imread('/mnt/A/CIKM2017/CIKM_datasets/test/sample_339/img_6.png',0).astype(np.float64)
    # pred_img = cv2.imread('/home/ices/img_6.png',0)
    # gt_img = cv2.imread('/mnt/A/CIKM2017/CIKM_datasets/test/sample_339/img_6.png',0)
    # print(pred_img)
    # print(gt_img)
    # print(pred_img.shape)
    # print(gt_img.shape)
    # for i in range(6,10)
    # compare_ssim()
    # print(calculate_ssim(pred_img,gt_img))
    # print(compare_ssim(pred_img,gt_img,255))
    # print(np.mean(np.square(pred_img-gt_img)))
    # test_model_list = [
    #     "CIKM_predrnn_test",
    #     "vertical_lstm_cikm_best",
    #     "horizontal_lstm_cikm_best",
    #     "CIKM_predrann_test",
    # ]
    # test_model_list = [
    #     # "CIKM_dec_ConvLSTM_test",
    #     "CIKM_convlstm_test",
    #     "CIKM_dec_ConvGRU_test",
    #     "CIKM_dec_TrajGRU_test",
    #     "CIKM_cdna_test",
    #     "dfn_test",
    #     "CIKM_predrnn_test",
    #     "CIKM_PredRNN_PP_test",
    #     "CIKM_MIM_test",
    #     "CIKM_e3d_lstm_test",
    #     "SA_CIKM",
    #     # "CIKM_inter_dst_predrnn_r2",
    #     # "CIKM_crev_net",
    #     "CIKM_phydnet",
    #     "CIKM_dec_PFST_ConvLSTM_test",
    #     # "CIKM_rprednet_x_test",
    #     # "CIKM_rprednet_m_test",
    #     # "CIKM_rprednet_h_test",
    #     # "CIKM_rprednet_test_",
    #
    #     "CIKM_predrann_test",
    # ]
    # test_model_list = [
    #     # "CIKM_convlstm_more3_test",
    #     # "CIKM_convlstm_more5_test",
    #     # "CIKM_convlstm_more7_test",
    #     # "CIKM_convlstm_more9_test",
    #     "CIKM_predrnn_test_more3",
    #     "CIKM_predrnn_test_more5",
    #     "CIKM_predrnn_test_more7",
    #     "CIKM_predrnn_test_more9",
    #     "CIKM_rprednet_x_test"
    # ]

    # data = []
    # for s in range(1,4001,1):
    #     if s%200==0:
    #         print(s)
    #
    #     data.append(flow_test(test_model_list,str(s)))
    # data = np.array(data)
    # print(data.shape)
    # print(np.mean(data,1))



    # plot_res(test_model_list)
    # all_t_test(test_model_list)
    # record_root_path = '/home/ices/Downloads/'
    # for model_name in test_model_list:
    #     cur_record_path = os.path.join(record_root_path,model_name+'.txt')
    #     record_info_extract(cur_record_path)
    #     break
    # seq_mse_res = {}
    # for model in test_model_list:
    #     seq_mse = sequence_mse(true_test_root, pred_root + model + '/', 'mse')
    #     seq_mse_res[model] = seq_mse
    # print(seq_mse_res)
    # plot(seq_mse_res,test_model_list)
    # validation_model_list = ['CIKM_dec_ST_ConvLSTM_validation','CIKM_dec_ConvGRU_validation', 'CIKM_dec_ConvLSTM_validation']

    # test_model_list = ['CIKM_ST_ConvLSTM_test', 'CIKM_ConvGRU_test', 'CIKM_ConvLSTM_test']
    # validation_model_list = ['CIKM_ST_ConvLSTM_validation', 'CIKM_TrajGRU_validation','CIKM_ConvGRU_validation',
    #                          'CIKM_ConvLSTM_validation']
    # validation_model_mse = {}
    # for model in validation_model_list:
    #     mse = eval_validation(true_validation_root, pred_root + model+'/', 'mse')
    #     validation_model_mse[model] = mse
    # print(validation_model_mse)

    test_model_list = [
        # "ConvGRU_test", #\0.7031 0.4857 0.1470 0.4453 0.7663 0.4092 0.0801  0.4186
        # "EqualWeight_ConvGRU_test",
        # "Uncertainty_ConvGRU_test",
        # "Correlation_ConvGRU_test_3",

        # "ConvLSTM_test",
        # "EqualWeight_ConvLSTM_test", #\0.7061 0.5047 0.1710 0.7642 0.7628 0.4176 0.0940 0.4253
        # "Uncertainty_ConvLSTM_test", #\0.7052 0.5166 0.1858 0.4692 0.7628 0.4279 0.1034 0.4313
        # "C4_ConvLSTM_test_", #\0.6741 0.4709 0.1832 0.4427 0.7402 0.4003 0.1017 0.4141
        
       
        # "ST_ConvLSTM_test", #\
        # "EqualWeight_ST_ConvLSTM_test", #\
        # "Uncertainty_ST_ConvLSTM_test",
        # "C_ST_ConvLSTM_test", #\
        
       
        # "CausalLSTM_test", #\
        # "EqualWeight_CausalLSTM_test", #\
        # "Uncertainty_CausalLSTM_test",
        # "C_CausalLSTM_test", #\
        
       
        # "SAConvLSTM_test", #\
        # "EqualWeight_SAConvLSTM_test", #\
        # "Uncerntainty_SAConvLSTM_test",
        # "C_SAConvLSTM_test", #\
        
       
        # "CMSLSTM_test", #\
        # "EqualWeight_CMSLSTM_test", #\
        # "Uncertainty_CMSLSTM_test",
        # "C_CMSLSTM_test", #\

        "ConvGRU_test", 
        "warp_ConvGRU_test",
        "reg_ConvGRU_test",
        "EqualWeight_ConvGRU_test",
        "Uncertainty_ConvGRU_test",
        "Correlation_ConvGRU_test_3",

        "ConvLSTM_test",
        "W2_ConvLSTM_test_", 
        "reg_ConvLSTM_test_2", 
        "EqualWeight_ConvLSTM_test", 
        "Uncertainty_ConvLSTM_test", 
        "C4_ConvLSTM_test_", 
        
       
        "ST_ConvLSTM_test", #\
        "warp_ST_ConvLSTM_test", #\
        "reg_ST_ConvLSTM_test",
        "EqualWeight_ST_ConvLSTM_test", #\
        "Uncertainty_ST_ConvLSTM_test",
        "C_ST_ConvLSTM_test", #\
        
       
        "CausalLSTM_test", #\
        "warp_CausalLSTM_test", #\
        "reg_CausalLSTM_test",
        "EqualWeight_CausalLSTM_test", #\
        "Uncertainty_CausalLSTM_test",
        "C_CausalLSTM_test", #\
        
       
        "SAConvLSTM_test", #\
        "warp_SAConvLSTM_test", #\
        "reg_SAConvLSTM_test__",
        "EqualWeight_SAConvLSTM_test", #\
        "Uncerntainty_SAConvLSTM_test",
        "C_SAConvLSTM_test", #\
        
       
        "CMSLSTM_test", #\
        "warp_CMSLSTM_test", #\
        "reg_CMSLSTM_test",
        "EqualWeight_CMSLSTM_test", #\
        "Uncertainty_CMSLSTM_test",
        "C_CMSLSTM_test", #\
    ]
    # test_model_list = [
    #     "CIKM_predrnn",
    #     "CIKM_predrann_s_test",
    #     "CIKM_predrann_t_test",
    #     "CIKM_predrann_test",
    # ]

    # test_model_mae = {}
    # for model in test_model_list:
    #     mae = eval_test(true_test_root, pred_root + model + '/', 'mae')
    #     test_model_mae[model] = mae
    #     print('mae model is:', model)
    #     print(test_model_mae[model])
    # print('*' * 80)

    # test_model_mse = {}
    # for model in test_model_list:
    #     mse = eval_test(true_test_root, pred_root + model + '/', 'mse')
    #     test_model_mse[model] = mse
    #     print('mse model is:', model)
    #     print(test_model_mse[model])
    # print('*' * 80)

    # test_model_bmae = {}
    # for model in test_model_list:
    #     mae = eval_test(true_test_root, pred_root + model + '/', 'b-mae')
    #     test_model_bmae[model] = mae
    #     print('b-mae model is:', model)
    #     print(test_model_bmae[model])
    # print('*' * 80)
    # test_model_bmse = {}
    # for model in test_model_list:
    #     mae = eval_test(true_test_root, pred_root + model + '/', 'b-mse')
    #     test_model_bmse[model] = mae
    #     print('b-mse model is:', model)
    #     print(test_model_bmse[model])
    # print('*' * 80)
    # test_model_ssim = {}
    # for id,model in enumerate(test_model_list):
    #     ssim = eval_test(true_test_root, pred_root + model + '/', 'ssim')
    #     test_model_ssim[model] = ssim
    #     print('ssim model is:', model)
    #     print(test_model_ssim[model])
    # print('*'*80)

    print('psnr')
    test_model_psnr = {}
    for id,model in enumerate(test_model_list):
        psnr = eval_test(true_test_root, pred_root + model + '/', 'psnr')
        test_model_psnr[model] = psnr
        print('psnr model is:', model)
        print(test_model_psnr[model])
    print('*'*80)
    #
    #
    #
    # # #
    # # print('ms-ssim')
    # # test_model_ms_ssim = {}
    # # for model in test_model_list:
    # #     ms_ssim = eval_test(true_test_root, pred_root + model + '/', 'ms-ssim')
    # #     test_model_ms_ssim[model] = ms_ssim
    # # print(test_model_ms_ssim)
    # # test_model_pcc = {}
    # # for model in test_model_list:
    # #     pcc = eval_test(sess,true_test_root, pred_root + model + '/', 'pcc')
    # #     test_model_pcc[model] = pcc
    # # print(test_model_pcc)
    #
    # # print(seq_mse)
    #
    # # ST_ConvLSTM_mse = [0.0024288102901641653, 0.003943319082303788, 0.005390102719733022, 0.006799462498499452, 0.008158839733144305, 0.00952125993827667, 0.010843842759111794, 0.012042934615375998, 0.013134818804459427, 0.014214933834487966]
    # # ConvGRU_mse = [0.002305122214578603, 0.003916686069469506, 0.005505802148131806, 0.007089604998223876, 0.00868397690187794, 0.01032106976072646, 0.01188595761911165, 0.013255351553541914, 0.01445867685882331, 0.01553596562302846]
    # # ConvLSTM_mse = [0.002295292251142598, 0.003928239543610289, 0.005540265110989367, 0.007083509935046095, 0.008604068437414753, 0.010130964369691355, 0.01155991896984051, 0.012890271273096004, 0.01410613953669963, 0.015222110682059793]
    # # dec_ST_ConvLSTM_mse = [0.002364268766264843, 0.003924998539975604, 0.005368746128718385, 0.006756065742258215, 0.008097049896226963, 0.009436994450309611, 0.010671709660275155, 0.011779709733296841, 0.012778628737465623, 0.013759537742738758]
    # # dec_ConvGRU_mse = [0.0024157174499055147, 0.00393797474564235, 0.005376405584767781, 0.006824714914899232, 0.008270984332273657, 0.009713952445783434, 0.011055264989359784, 0.01226929799222671, 0.013360691534066063, 0.014434498674891074]
    # # dec_ConvLSTM_mse = [0.002379041242172434, 0.003937165613700017, 0.0054181525517010415, 0.006915119287572452, 0.008500685820710714, 0.010079977763036367, 0.011591412258920172, 0.012939707933073806, 0.014157152997035155, 0.015297246081059711]
    #
    # # ST_ConvLSTM_mse = [0.002302415528975871, 0.0039257768223305905, 0.005485519943306372, 0.00702758381395779, 0.008578271590690746, 0.010112517563517031, 0.011512441961131117, 0.012731947152129578, 0.013820672234536688, 0.014873968772197259]
    # # ConvGRU_mse = [0.002292289455213222, 0.003897478523168502, 0.005540548081439738, 0.007177185952796208, 0.008780076390316026, 0.010361579176989835, 0.011876730150050207, 0.013251351497496216, 0.014447874331055573, 0.015510209560919976]
    # # ConvLSTM_mse = [0.002332684729696666, 0.0039732200092421404, 0.00555670658428221, 0.007159864549257691, 0.008815937865809247, 0.010477123910687624, 0.012064420173284816, 0.013430322390984656, 0.014589850207026757, 0.015688560969136234]
    # # dec_ST_ConvLSTM_mse = [0.002344160946124248, 0.003938462443034041, 0.005453599778101306, 0.006914397735837156, 0.00835189992823507, 0.009795115892367904, 0.011185708106666425, 0.012469912361173556, 0.013657610348433082, 0.014840640853370132]
    # # dec_ConvGRU_mse = [0.0026988081292238348, 0.004471600576382116, 0.006056587053779367, 0.007522618366359893, 0.008887108978367905, 0.01020160230613692, 0.011468532416663948, 0.012602082935030921, 0.01361446809602785, 0.014613186329894234]
    # # dec_ConvLSTM_mse = [0.0023624405927522504, 0.003915410714364953, 0.005393563923598776, 0.006865441179938898, 0.008360895798168712, 0.009896392613007265, 0.01136580711356146, 0.01268309452279027, 0.013799309661371807, 0.014857410416574566]
    # # print(
    # # 'ConvGRU:'+format(np.mean(np.array(ConvGRU_mse))*100,'.4f')
    # # , 'ConvLSTM:' + format(np.mean(np.array(ConvLSTM_mse)) * 100, '.4f')
    # # , 'ST_ConvLSTM:'+format(np.mean(np.array(ST_ConvLSTM_mse))*100,'.4f')
    # # , 'dec_ConvGRU:' + format(np.mean(np.array(dec_ConvGRU_mse)) * 100, '.4f')
    # # , 'dec_ConvLSTM:' + format(np.mean(np.array(dec_ConvLSTM_mse)) * 100, '.4f')
    # # , 'dec_ST_ConvLSTM:' + format(np.mean(np.array(dec_ST_ConvLSTM_mse)) * 100, '.4f')
    # # )
    # #
    # #
    # # plot(
    # #     [ConvGRU_mse,ConvLSTM_mse,ST_ConvLSTM_mse,dec_ConvGRU_mse,dec_ConvLSTM_mse,dec_ST_ConvLSTM_mse],
    # #     [
    # #         'ConvGRU:'+format(np.mean(np.array(ConvGRU_mse))*100,'.4f')
    # #         , 'ConvLSTM:' + format(np.mean(np.array(ConvLSTM_mse)) * 100, '.4f')
    # #         , 'ST_ConvLSTM:'+format(np.mean(np.array(ST_ConvLSTM_mse))*100,'.4f')
    # #         , 'dec_ConvGRU:' + format(np.mean(np.array(dec_ConvGRU_mse)) * 100, '.4f')
    # #         , 'dec_ConvLSTM:' + format(np.mean(np.array(dec_ConvLSTM_mse)) * 100, '.4f')
    # #         , 'dec_ST_ConvLSTM:' + format(np.mean(np.array(dec_ST_ConvLSTM_mse)) * 100, '.4f')
    # #
    # #     ]
    # # )
    #
    # # mse = eval_test(true_test_root, pred_root + 'CIKM_cl_dec_ST_ConvLSTM_test/', 'mse')
    # # print('CIKM_cl_dec_ST_ConvLSTM_test mse is:', mse)
    # # mse = eval_test(true_test_root, pred_root + 'CIKM_dec_TrajGRU_test/', 'mse')
    # # print('CIKM_dec_TrajGRU_test mse is:', mse)
    # # mse = eval_test(true_test_root, pred_root + 'CIKM_Conv2DLSTM_test/', 'mse')
    # # print('CIKM_Conv2DLSTM_test mse is:', mse)
    # # mse = eval_test(true_test_root, pred_root + 'CIKM_cl_dec_ST_ConvLSTM_test/', 'mse')
    # # print('CIKM_cl_dec_ST_ConvLSTM_test mse is:', mse)
    #
    # # mse = eval_validation(true_validation_root, pred_root + 'CIKM_ConvGRU_validation/', 'mse')
    # # print('CIKM_ConvGRU_validation mse is:', mse)
    # # mae = eval_test(true_test_root, pred_root + 'CIKM_ST_ConvLSTM_test/', 'mae')
    # # print('mae is:', mae)
    # # ssim = eval_test(true_test_root, pred_root + 'CIKM_ConvGRU_test/', 'ssim')
    # # print('ssim is:', ssim)
    # # pcc = eval_test(true_test_root, pred_root + 'CIKM_ConvGRU_test/', 'pcc')
    # # print('pcc is:', pcc)