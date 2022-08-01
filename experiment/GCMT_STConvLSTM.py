import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)
import yaml
import torch
import torch.optim as optim
from util.utils import *
import math
from util.LossFunction import *
from experiment.Encode_Decode import *
from data_provider.cikm_radar import *



def to_device(data):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return data.to(device,non_blocking = True)

class sequence_model(object):
    def __init__(self,
                 name,
                 encoder_decoder_model,
                 info,
                 ):
        self.info = info
        self.name = name
        self.data_root = info['DATA']['ROOT_PATH']
        self.encoder_decoder_model = encoder_decoder_model
        train_data = SeqRadar(
            data_type='train',
            data_root=self.data_root,
        )
        valid_data = SeqRadar(
            data_type='validation',
            data_root=self.data_root
        )
        test_data = SeqRadar(
            data_type='test',
            data_root=self.data_root
        )
        self.train_loader = DataLoader(train_data,
                                  num_workers=2,
                                  batch_size=info['TRAIN']['BATCH_SIZE'],
                                  shuffle=False,
                                  drop_last=False,
                                  pin_memory=False)
        self.valid_loader = DataLoader(valid_data,
                                  num_workers=1,
                                  batch_size=info['TRAIN']['BATCH_SIZE'],
                                  shuffle=False,
                                  drop_last=False,
                                  pin_memory=False)
        self.test_loader = DataLoader(test_data,
                                 num_workers=1,
                                 batch_size=info['TRAIN']['BATCH_SIZE'],
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=False)
        self.validation_save_root = self.info['VALID_SAVE_ROOT']
        self.test_save_root = self.info['TEST_SAVE_ROOT']
        self.batch_size = self.info['TRAIN']['BATCH_SIZE']
        self.criterion = AutomaticWeightedLoss(3).cuda()
        self.optimizer = optim.Adam([
            {'params': self.encoder_decoder_model.parameters()},
            {'params': self.criterion.parameters(), 'weight_decay': 0}
        ],lr=self.info['TRAIN']['LEARNING_RATE'], eps = 1e-5)


    def train(self):
        tolerate_iter = 0
        best_mse = math.inf
        if torch.cuda.is_available():
            self.encoder_decoder_model = self.encoder_decoder_model.cuda()
        self.encoder_decoder_model.train()
        
        criterion1 = torch.nn.MSELoss(reduce = False).cuda()
        step = 0
        for train_iter in range(self.info['TRAIN']['EPOCHES']):
            
            for i_batch, batch_data in enumerate(self.train_loader):
                frame_dat = batch_data[0].cuda()
                seg_label = batch_data[1].cuda()
                in_frame_dat = frame_dat[:,:5]
                target_frame_dat = frame_dat[:,5:]
                target_seg_label = seg_label[:,5:]
                self.optimizer.zero_grad()
                output, warp_out, reg_out, hidden = self.encoder_decoder_model(in_frame_dat)

                l_det = criterion1(input=output, target=target_frame_dat)
                l_warp = criterion1(input=warp_out, target=target_frame_dat)
                l_reg = criterion1(input=reg_out, target=target_frame_dat)

                w_pred = get_st_reg_weights(self.encoder_decoder_model.models[1], hidden, target_frame_dat)
                w_warp = get_st_warp_weights(self.encoder_decoder_model.models[1], hidden, target_frame_dat, in_frame_dat[:, -1])
                w_reg = get_st_regression_weights(self.encoder_decoder_model.models[1], hidden, target_frame_dat)
                c_warp = torch.abs(compute_rank_correlation(torch.reshape(w_warp, (4*10, 64*51*51)), torch.reshape(w_pred, (4*10, 64*51*51))))
                c_reg = torch.abs(compute_rank_correlation(torch.reshape(w_reg, (4*10, 64*51*51)), torch.reshape(w_pred, (4*10, 64*51*51))))
                c1 = c_warp / (c_reg + c_warp)
                c2 = c_reg / (c_reg + c_warp)
                c1 = torch.reshape(c1, (4, 10, 1, 1, 1))
                c2 = torch.reshape(c2, (4, 10, 1, 1, 1))

                loss  = (l_det + l_reg * c2 + c1 * l_warp).mean()
                loss.backward()
                self.optimizer.step()

                step = step+1
                if (step+1)%self.info['TRAIN']['DISTPLAY_STEP'] == 0:
                    print('Train iter is:',(train_iter+1),'current loss is:',float(loss.cpu().data.numpy()))

            
            cur_loss = self.validation()
            print('validation loss is :', cur_loss)
            if cur_loss < best_mse:
                best_mse = cur_loss
                tolerate_iter = 0
                self.save_model()
            else:
                tolerate_iter += 1
                if tolerate_iter == self.info['TRAIN']['LOSS_LIMIT']:
                    print('the best validation loss is:', best_mse)
                    self.load_model()
                    test_loss = self.test()
                    print('the best test loss is:', test_loss)
                    break

    def validation(self):
        self.encoder_decoder_model.eval()
        if torch.cuda.is_available():
            self.encoder_decoder_model = self.encoder_decoder_model.cuda()
        loss = 0
        count = 0
        for i_batch, batch_data in enumerate(self.valid_loader):

            frame_dat = batch_data[0].cuda()

            in_frame_dat = frame_dat[:, :5]
            target_frame_dat = frame_dat[:, 5:]
            output_frames,_,_,_ = self.encoder_decoder_model(in_frame_dat)
            output_frames = denormalization(output_frames.data.cpu().numpy(), 255.0).astype(np.float32) / 255.0
            target_frames = denormalization(target_frame_dat.data.cpu().numpy(), 255.0).astype(np.float32) / 255.0
            current_loss = np.mean(np.square(output_frames - target_frames))
            loss = current_loss + loss
            count = count + 1

        loss = loss / count
        return loss

    def test(self, is_save=True):
        batch_size = self.batch_size
        assert self.test_save_root is not None
        test_save_root = self.test_save_root

        clean_fold(test_save_root)
        self.encoder_decoder_model.eval()
        if torch.cuda.is_available():
            self.encoder_decoder_model = self.encoder_decoder_model.cuda()
        loss = 0
        count = 0
        for i_batch, batch_data in enumerate(self.test_loader):
            frame_dat = batch_data[0][0].cuda()
            cur_fold = batch_data[1]
            in_frame_dat = frame_dat[:, :5]
            target_frame_dat = frame_dat[:, 5:]
            output_frames,_,_,_ = self.encoder_decoder_model(in_frame_dat)
            output_frames = denormalization(output_frames.data.cpu().numpy(), 255.0).astype(np.float32) / 255.0
            target_frames = denormalization(target_frame_dat.data.cpu().numpy(), 255.0).astype(np.float32) / 255.0
            current_loss = np.mean(np.square(output_frames - target_frames))
            output_frames = (output_frames * 255.0).astype(np.uint8)
            loss = current_loss + loss
            count = count + 1

            for bat_ind in range(batch_size):
                cur_batch_data = output_frames[bat_ind, :, 0, :, :, ]

                cur_sample_fold = os.path.join(test_save_root, cur_fold[bat_ind])

                if not os.path.exists(cur_sample_fold):
                    os.mkdir(cur_sample_fold)

                for t in range(10):
                    cur_save_path = os.path.join(cur_sample_fold, 'img_' + str(t + 6) + '.png')

                    cur_img = cur_batch_data[t]

                    cv2.imwrite(cur_save_path, cur_img)

        loss = loss / count
        print('test loss is :', str(loss))
        return loss

    def save_model(self):
        if not os.path.exists(os.path.split(self.info['MODEL_SAVE_PATH'])[0]):
            os.makedirs(os.path.split(configuration['MODEL_SAVE_PATH'])[0])
        torch.save(
            self.encoder_decoder_model,
            self.info['MODEL_SAVE_PATH']
        )
        print('model saved')

    def load_model(self):
        if not os.path.exists(os.path.split(configuration['MODEL_SAVE_PATH'])[0]):
            raise ('there are not model in ', os.path.split(configuration['MODEL_SAVE_PATH'])[0])
        self.encoder_decoder_model = torch.load(
            self.info['MODEL_SAVE_PATH']
        )
        print('model has been loaded')

if __name__ == '__main__':

    path = 'configs/GCMT_STConvLSTM.yml'
    f = open(path)
    configuration = yaml.safe_load(f)

    encode_shape = configuration['MODEL_NETS']['ENSHAPE']
    decode_shape = configuration['MODEL_NETS']['DESHAPE']

    encode_conv_rnn_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['ENCODE_CELLS']):
        param = get_cell_param(cell)
        param['m_channels'] = configuration['MODEL_NETS']['m_channels'][idx]
        encode_conv_rnn_cells.append(param)

    downsample_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['DOWNSAMPLE_CONVS']):
        if idx == len(configuration['MODEL_NETS']['DOWNSAMPLE_CONVS']) - 1:
            downsample_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['ENCODE_PADDING'][idx], activate='tanh'))
        else:
            downsample_cells.append(get_conv_param(cell, padding=configuration['MODEL_NETS']['ENCODE_PADDING'][idx]))

    decode_conv_rnn_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['DECODE_CELLS']):
        param = get_cell_param(cell)
        param['m_channels'] = configuration['MODEL_NETS']['m_channels'][idx]
        decode_conv_rnn_cells.append(param)

    upsample_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['UPSAMPLE_CONVS']):
        if idx == len(configuration['MODEL_NETS']['UPSAMPLE_CONVS']) - 1:
            upsample_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['DECODE_PADDING'][idx], activate='tanh'))
        else:
            upsample_cells.append(get_conv_param(cell, padding=configuration['MODEL_NETS']['DECODE_PADDING'][idx]))

    output_conv_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['OUTPUT_CONV']):
        if idx == len(configuration['MODEL_NETS']['OUTPUT_CONV']) - 1:
            output_conv_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['OUTPUT_PADDING'][idx], activate='tanh'))
        else:
            output_conv_cells.append(get_conv_param(cell, padding=configuration['MODEL_NETS']['OUTPUT_PADDING'][idx]))

    reg_output_conv_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['REG_OUTPUT_CONV']):
        if idx == len(configuration['MODEL_NETS']['REG_OUTPUT_CONV']) - 1:
            reg_output_conv_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['REG_OUTPUT_PADDING'][idx], activate='tanh'))
        else:
            reg_output_conv_cells.append(get_conv_param(cell, padding=configuration['MODEL_NETS']['OUTPUT_PADDING'][idx]))

    encode_conv_m_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['M_ENCODE']):
        if idx == len(configuration['MODEL_NETS']['M_ENCODE']) - 1:
            encode_conv_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_ENCODE_PADDING'][idx], activate='tanh'))
        else:
            encode_conv_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_ENCODE_PADDING'][idx]))

    decode_conv_m_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['M_DECODE']):
        if idx == len(configuration['MODEL_NETS']['M_DECODE']) - 1:
            decode_conv_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_DECODE_PADDING'][idx], activate='tanh'))
        else:
            decode_conv_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_DECODE_PADDING'][idx]))

    encode_conv_reset_m_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['M_DECODE']):
        if idx == len(configuration['MODEL_NETS']['M_DECODE']) - 1:
            encode_conv_reset_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_DECODE_PADDING'][idx], activate='tanh'))
        else:
            encode_conv_reset_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_DECODE_PADDING'][idx]))

    decode_conv_reset_m_cells = []
    for idx, cell in enumerate(configuration['MODEL_NETS']['M_ENCODE']):
        if idx == len(configuration['MODEL_NETS']['M_ENCODE']) - 1:
            decode_conv_reset_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_ENCODE_PADDING'][idx], activate='tanh'))
        else:
            decode_conv_reset_m_cells.append(
                get_conv_param(cell, padding=configuration['MODEL_NETS']['M_ENCODE_PADDING'][idx]))

    encoder = Encoder_ST_ConvLSTM(
        encode_shape=encode_shape,
        conv_rnn_cells=encode_conv_rnn_cells,
        conv_cells=downsample_cells,
        conv_m_cells=encode_conv_m_cells,
        conv_reset_m_cells=encode_conv_reset_m_cells,
        info=configuration
    ).cuda()

    decoder = Decoder_ST_ConvLSTM(
        decode_shape=decode_shape,
        conv_rnn_cells=decode_conv_rnn_cells,
        conv_cells=upsample_cells,
        conv_m_cells=decode_conv_m_cells,
        conv_reset_m_cells=decode_conv_reset_m_cells,
        output_cells=output_conv_cells,
        reg_output_cells = reg_output_conv_cells,
        info=configuration
    )

    encoder_decoder_model = Encode_Decode_ST_ConvLSTM(
        encoder = encoder,
        decoder = decoder,
        info = configuration
    )

    model = sequence_model(
        name=configuration['NAME'],
        encoder_decoder_model = encoder_decoder_model,
        info = configuration
    )
    # model.load_model()
    # model.test()
    model.train()
    # model.validation()

