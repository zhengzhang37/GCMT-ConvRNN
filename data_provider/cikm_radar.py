import os
from scipy.misc import *
import random
import numpy as np
import torch
# from scipy.misc import imread
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
# from Gaussian_dataprocess import *



class SeqRadar_(Dataset):

    def __init__(self,data_type,data_root='train'):
        self.data_type = data_type
        self.data_root = data_root # [train , valid , test]
        self.dirs = os.listdir("{}".format(os.path.join(self.data_root,self.data_type)))


    def __len__(self):
        return len(self.dirs)

    def read_seg_img(self,path):
        img = cv2.imread(path)
        img = img.astype(np.float)
        dBZ_img = img*95/255 - 10
        dBZ_img[dBZ_img < 5] = 4
        dBZ_img[dBZ_img >= 40] = 3
        dBZ_img[dBZ_img >= 20] = 2
        dBZ_img[dBZ_img >= 5] = 1
        dBZ_img[dBZ_img == 4] = 0
        dBZ_img = dBZ_img.astype(np.float)

        h,w = dBZ_img.shape
        target_dBZ_img = np.zeros((4,h,w))
        for c in range(4):
            target_dBZ_img[c][dBZ_img==c]=1
        return target_dBZ_img

    def __getitem__(self, index):
        cur_fold = os.path.join(self.data_root,self.data_type,self.dirs[index])
        files = os.listdir(cur_fold)
        files.sort()
        imgs = []
        seg_imgs = []
        for i in range(len(files)):
            file = 'img_'+str(i+1)+'.png'
            img_path = os.path.join(cur_fold,file)
            img = cv2.imread(img_path)[np.newaxis,:,:,]

            seg_img = self.read_seg_img(img_path)
            seg_imgs.append(seg_img)
            imgs.append(img)
        imgs = np.array(imgs)
        seg_imgs = np.array(seg_imgs)
        imgs = imgs.astype(np.float32)/255.0
        if self.data_type == 'test':
            return [imgs,seg_imgs],self.dirs[index]
        else:
            return [imgs,seg_imgs]

class SeqRadar(Dataset):

    def __init__(self,data_type, data_root='train'):
        self.data_type = data_type
        self.data_root = data_root # [train , valid , test]
        self.dirs = os.listdir("{}".format(os.path.join(self.data_root,self.data_type)))


    def __len__(self):
        return len(self.dirs)

    def read_seg_img(self,path):
        img = cv2.imread(path, 0)
        img = img.astype(np.float)
        dBZ_img = img*95/255 - 10
        dBZ_img[dBZ_img < 5] = 4
        dBZ_img[dBZ_img >= 40] = 3
        dBZ_img[dBZ_img >= 20] = 2
        dBZ_img[dBZ_img >= 5] = 1
        dBZ_img[dBZ_img == 4] = 0
        dBZ_img = dBZ_img.astype(np.float)

        h,w = dBZ_img.shape
        target_dBZ_img = np.zeros((4,h,w))
        for c in range(4):
            target_dBZ_img[c][dBZ_img==c]=1
        return dBZ_img

    def __getitem__(self, index):
        cur_fold = os.path.join(self.data_root,self.data_type,self.dirs[index])
        files = os.listdir(cur_fold)
        files.sort()
        imgs = []
        seg_imgs = []
        for i in range(len(files)):
            file = 'img_'+str(i+1)+'.png'
            img_path = os.path.join(cur_fold,file)
            img = cv2.imread(img_path, 0)[np.newaxis,:,:,]
            seg_img = self.read_seg_img(img_path)
            seg_imgs.append(seg_img)
            imgs.append(img)

        imgs = np.array(imgs)
        seg_imgs = np.array(seg_imgs)
        imgs = 2*(imgs.astype(np.float32)/255.0 - 0.5)
        if self.data_type == 'test':
            return [imgs,seg_imgs],self.dirs[index]
        else:
            return [imgs,seg_imgs]

class Radar(Dataset):

    def __init__(self,data_type,data_root='train'):
        self.data_type = data_type
        self.data_root = data_root # [train , valid , test]
        self.dirs = os.listdir("{}".format(os.path.join(self.data_root,self.data_type)))

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, index):
        cur_fold = os.path.join(self.data_root,self.data_type,self.dirs[index])
        files = os.listdir(cur_fold)
        files.sort()
        imgs = []
        for i in range(len(files)):
            file = 'img_'+str(i+1)+'.png'
            img_path = os.path.join(cur_fold,file)
            img = cv2.imread(img_path)[np.newaxis,:,:,]
            imgs.append(img)
        imgs = np.array(imgs)
        # imgs = imgs.astype(np.float32)/255.0
        imgs = 2 * (imgs.astype(np.float32) / 255.0 - 0.5)
        if self.data_type == 'test':
            return imgs,self.dirs[index]
        else:
            return imgs

class Radar_(Dataset):

    def __init__(self,data_type,data_root='train'):
        self.data_type = data_type
        self.data_root = data_root # [train , valid , test]
        self.dirs = os.listdir("{}".format(os.path.join(self.data_root,self.data_type)))

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, index):
        cur_fold = os.path.join(self.data_root,self.data_type,self.dirs[index])
        files = os.listdir(cur_fold)
        files.sort()
        imgs = []
        for i in range(len(files)):
            file = 'img_'+str(i+1)+'.png'
            img_path = os.path.join(cur_fold,file)
            img = imread(img_path)[:,:,np.newaxis]
            imgs.append(img)
        imgs = np.array(imgs)
        # imgs = imgs.astype(np.float32)/255.0

        if self.data_type == 'test':
            return imgs,self.dirs[index]
        else:
            return imgs
# class Radar(Dataset):
#
#     def __init__(self,data_type,data_root='train'):
#         self.data_type = data_type
#         self.data_root = data_root # [train , valid , test]
#         self.dirs = os.listdir("{}".format(os.path.join(self.data_root,self.data_type)))
#
#
#     def __len__(self):
#         return len(self.dirs)
#
#     def __getitem__(self, index):
#         cur_fold = os.path.join(self.data_root,self.data_type,self.dirs[index])
#         files = os.listdir(cur_fold)
#         files.sort()
#         imgs = []
#         for i in range(len(files)):
#             file = 'img_'+str(i+1)+'.png'
#             img_path = os.path.join(cur_fold,file)
#             img = imread(img_path)[np.newaxis,:,:,]
#             imgs.append(img)
#         imgs = np.array(imgs)
#
#         # if np.sum(imgs[-1])-np.sum(imgs[4])>0:
#         #     label = 1
#         # else:
#         #     label = 0
#         labels = []
#         for i in range(10):
#             label = (5 * np.sum(imgs[5 + i]) / np.sum(imgs[4])+
#                      4 * np.sum(imgs[5 + i]) / np.sum(imgs[3])+
#                      3 * np.sum(imgs[5 + i]) / np.sum(imgs[2])+
#                      2 * np.sum(imgs[5 + i]) / np.sum(imgs[1])+
#                      1 * np.sum(imgs[5 + i]) / np.sum(imgs[0]))/(5+4+3+2+1)
#             labels.append(label)
#         labels = np.array(labels).astype(np.float32)
#         imgs = 2*(imgs.astype(np.float32)/255.0 - 0.5)
#         if self.data_type == 'test':
#             return [imgs,labels],self.dirs[index]
#         else:
#             return [imgs,labels]


if __name__ == '__main__':

    batch_size = 4
    data_root = '/mnt/A/CIKM2017/CIKM_datasets/'
    train_data = SeqRadar(
        data_type='train',
        data_root=data_root,
    )
    valid_data = Radar(
        data_type='validation',
        data_root=data_root
    )
    test_data = Radar(
        data_type='test',
        data_root=data_root
    )
    train_loader = DataLoader(train_data,
                              num_workers=2,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=False)
    valid_loader = DataLoader(valid_data,
                              num_workers=1,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=False)
    test_loader = DataLoader(test_data,
                             num_workers = 1,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=False)
#     save_fold = '../example_result'
#     if os.path.exists(save_fold):
#         pass
#     else:
#         os.mkdir(save_fold)
#
    for i_batch,batch_data in enumerate(train_loader):
        # batch_data = batch_data.cuda()
        print('train',str(i_batch),batch_data[0].numpy().shape,batch_data[1].numpy().shape)
        img1 = batch_data[0].numpy()[0,0,0]
        img2 = batch_data[1].numpy()[0,0,0]
        print(img1.shape)
        print(img2.shape)
        # imsave('img.png',img1)
        # print(np.max(img2),np.max(img1))
        # img1 = img1 * 255.0
        # img1 = img1 * 95 / 255 - 10
        # print(np.max(img1))
        # img2[img2 == 1] = 75
        # img2[img2 == 2] = 150
        # img2[img2 == 3] = 250
        # imsave('seg_img.png',img2)
        break
    # for i_batch, batch_data in enumerate(valid_loader):
    #     # batch_data = batch_data.cuda()
    #     print('valid',str(i_batch), batch_data.numpy().shape)
    # for i_batch, batch_data in enumerate(test_loader):
    #     # batch_data = batch_data.cuda()
    #     print('test',str(i_batch), batch_data.numpy().shape)
        # batch_data = batch_data.detach().cpu().numpy()[0,:,0,:,:]
        # t_length = batch_data.shape[0]
        # for t in range(t_length):
        #     cur_img = batch_data[t]
        #     cur_path = os.path.join(save_fold,'img_'+str(t+1)+'.png')
        #     imsave(cur_path,cur_img)
        # print(batch_data.shape,str(i_batch))

    # train data check
    # train_data_root = os.path.join(data_root,'test')
    # folds = os.listdir(train_data_root)
    # folds.sort()
    # for fold in folds:
    #     cur_fold_path = os.path.join(train_data_root,fold)
    #     print(cur_fold_path)
    #     files = os.listdir(cur_fold_path)
    #     for file in files:
    #         cur_img_path = os.path.join(cur_fold_path,file)
    #         img = imread(cur_img_path)[:,:,np.newaxis]
