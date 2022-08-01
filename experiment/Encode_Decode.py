from model.ConvLSTM import *
from model.ConvGRU import *
import model.ConvSeq as conv_seq
import model.Conv as conv
from model.SpatioTemporalLSTMCell import *
from model.Conv import *
from model.CausalLSTMCell import *
from model.SAConvLSTMCell import *
from model.CMSLSTMCell import *
class Encode_Decode_ConvLSTM(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 info,
                 ):
        super(Encode_Decode_ConvLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.info = info
        self.models = nn.ModuleList([self.encoder,self.decoder])

    def forward(self, input):
        in_decode_frame_dat = Variable(torch.zeros(
            self.info['TRAIN']['BATCH_SIZE'],
            self.info['DATA']['OUTPUT_SEQ_LEN'],
            self.info['MODEL_NETS']['ENCODE_CELLS'][-1][1],
            self.info['MODEL_NETS']['DESHAPE'][0],
            self.info['MODEL_NETS']['DESHAPE'][0],
        ).cuda())
        encode_states = self.models[0](input)
        output, warp_output, reg_output, current_conv_output = self.models[1](in_decode_frame_dat, input, encode_states)
        return output, warp_output, reg_output, current_conv_output

class Decoder_ConvLSTM(nn.Module):
    def __init__(self,
                 conv_rnn_cells,
                 conv_cells,
                 output_cells,
                 warp_output_cells,
                 reg_output_cells,
                 ):
        super(Decoder_ConvLSTM, self).__init__()
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.models = []
        self.output_cells = output_cells
        self.warp_output_cells = warp_output_cells
        self.reg_output_cells = reg_output_cells
        self.layer_num = len(self.conv_rnn_cells)

        for idx in range(self.layer_num):
            self.models.append(
                ConvLSTM(
                    cell_param=self.conv_rnn_cells[idx],
                    return_sequence=True,
                    return_state=False,
                )
            )
            self.models.append(
                conv_seq.DeConv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )


        self.models = nn.ModuleList(self.models)
        self.out_models = []
        for output_cell in output_cells:
            self.out_models.append(
                conv_seq.Conv2D(
                    cell_param=output_cell
                ).cuda()
            )
        self.out_models = nn.ModuleList(self.out_models)

        self.reg_models = []
        
        for reg_output_cell in reg_output_cells:
            self.reg_models.append(
                conv_seq.DeConv2D(
                    cell_param=self.conv_cells[2]
                ).cuda()
            )

            self.reg_models.append(
                conv_seq.Conv2D(
                    cell_param=reg_output_cell
                ).cuda()
            )
        self.reg_models = nn.ModuleList(self.reg_models)
        self.flow_model = nn.Conv2d(self.conv_cells[-1]['out_channel'], 2, kernel_size=1, stride=1, padding=0, bias=True)

    def get_warped_images(self, flows, last_frame):
        wrap_preds = []
        for j in range(len(flows[0])):
            if j == 0:
                h = last_frame
            flow = flows[:, j, 0:2]
            wrap_pred = self.wrap(h, -flow)
            h = wrap_pred
            wrap_preds.append(wrap_pred)
        return wrap_preds

    def wrap(self, input, flow):
        B, C, H, W = input.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1).cuda()
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W).cuda()
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().cuda()
        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = torch.nn.functional.grid_sample(input, vgrid)
        return output
    
    def forward(self, input, origin_input, state = None):
        assert state is not None
        last_frame = origin_input[:,-1]
        multi_layers_hidden_states = []
        warp_features = []
        
        
        for layer_idx in range(self.layer_num-1):
            current_conv_rnn_output = self.models[2*layer_idx](input,state[self.layer_num-1-layer_idx])
            multi_layers_hidden_states.append(current_conv_rnn_output)
            current_conv_output = self.models[2*layer_idx+1](current_conv_rnn_output)
            if layer_idx == self.layer_num-1:
                pass
            else:
                input = current_conv_output
                
        current_conv_rnn_output = self.models[2*2](input,state[self.layer_num-1-2])


        output = current_conv_rnn_output.clone()
        output = self.models[2*2+1](output)
        for out_layer_idx in range(len(self.out_models)):
            output = self.out_models[out_layer_idx](output)


        warp_input_seqs = current_conv_rnn_output.clone()
        warp_input_seqs = self.models[2*2+1](warp_input_seqs)
        warp_outputs = []
        for out_layer_idx in range(10):
            warp_features.append(self.flow_model(warp_input_seqs[:, out_layer_idx]))
        warp_features = torch.stack(warp_features, 1)
        warp_outputs = self.get_warped_images(warp_features, last_frame)
        warp_outputs = torch.stack(warp_outputs,1)  


        reg_output = current_conv_rnn_output.clone()
        for out_layer_idx in range(len(self.reg_models)):
            reg_output = self.reg_models[out_layer_idx](reg_output)
        return output, warp_outputs, reg_output, current_conv_rnn_output

class Encoder_ConvLSTM(nn.Module):
    def __init__(self,
                 conv_rnn_cells,
                 conv_cells,
    ):
        super(Encoder_ConvLSTM, self).__init__()

        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        for idx in range(self.layer_num):
            self.models.append(
                conv_seq.Conv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )
            self.models.append(
                ConvLSTM(
                    cell_param=self.conv_rnn_cells[idx],
                    return_sequence=True,
                    return_state=True,
                )
            )

        self.models = nn.ModuleList(self.models)

    def forward(self, input,state = None):
        encode_state = []
        for layer_idx in range(self.layer_num):
            current_conv_output = self.models[2*layer_idx](input)
            current_conv_rnn_output,state = self.models[2*layer_idx+1](current_conv_output)
            encode_state.append(state)
            if layer_idx == self.layer_num-1:
                pass
            else:
                input = current_conv_rnn_output
        return encode_state

class Encoder_ConvGRU(nn.Module):
    def __init__(self,
                 conv_rnn_cells,
                 conv_cells,
    ):
        super(Encoder_ConvGRU, self).__init__()

        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        for idx in range(self.layer_num):
            self.models.append(
                conv_seq.Conv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )
            self.models.append(
                ConvGRU(
                    cell_param=self.conv_rnn_cells[idx],
                    return_sequence=True,
                    return_state=True,
                )
            )

        self.models = nn.ModuleList(self.models)

    def forward(self, input,state = None):
        encode_state = []
        for layer_idx in range(self.layer_num):
            current_conv_output = self.models[2*layer_idx](input)
            current_conv_rnn_output,state = self.models[2*layer_idx+1](current_conv_output)
            encode_state.append(state)
            if layer_idx == self.layer_num-1:
                pass
            else:
                input = current_conv_rnn_output
        return encode_state

class Decoder_ConvGRU(nn.Module):
    def __init__(self,
                 conv_rnn_cells,
                 conv_cells,
                 output_cells,
                 warp_output_cells,
                 reg_output_cells,
                 ):
        super(Decoder_ConvGRU, self).__init__()
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.models = []
        self.output_cells = output_cells
        self.warp_output_cells = warp_output_cells
        self.reg_output_cells = reg_output_cells
        self.layer_num = len(self.conv_rnn_cells)

        for idx in range(self.layer_num):
            self.models.append(
                ConvGRU(
                    cell_param=self.conv_rnn_cells[idx],
                    return_sequence=True,
                    return_state=False,
                )
            )
            self.models.append(
                conv_seq.DeConv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )


        self.models = nn.ModuleList(self.models)


        self.out_models = []
        for output_cell in output_cells:
            self.out_models.append(
                conv_seq.Conv2D(
                    cell_param=output_cell
                ).cuda()
            )
        self.out_models = nn.ModuleList(self.out_models)

        self.reg_models = []
        
        for reg_output_cell in reg_output_cells:
            self.reg_models.append(
                conv_seq.DeConv2D(
                    cell_param=self.conv_cells[2]
                ).cuda()
            )
            self.reg_models.append(
                conv_seq.Conv2D(
                    cell_param=reg_output_cell
                ).cuda()
            )
        self.reg_models = nn.ModuleList(self.reg_models)
        self.flow_model = nn.Conv2d(self.conv_cells[-1]['out_channel'], 2, kernel_size=1, stride=1, padding=0, bias=True)

    def get_warped_images(self, flows, last_frame):
        wrap_preds = []
        for j in range(len(flows[0])):
            if j == 0:
                h = last_frame
            flow = flows[:, j, 0:2]
            wrap_pred = self.wrap(h, -flow)
            h = wrap_pred
            wrap_preds.append(wrap_pred)
            
        return wrap_preds

    def wrap(self, input, flow):
        B, C, H, W = input.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1).cuda()
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W).cuda()
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().cuda()
        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = torch.nn.functional.grid_sample(input, vgrid)
        return output
    
    def forward(self, input, origin_input, state = None):
        last_frame = origin_input[:,-1]
        assert state is not None
        multi_layers_hidden_states = []
        warp_features = []
        for layer_idx in range(self.layer_num-1):
            current_conv_rnn_output = self.models[2*layer_idx](input,state[self.layer_num-1-layer_idx])
            multi_layers_hidden_states.append(current_conv_rnn_output)

            current_conv_output = self.models[2*layer_idx+1](current_conv_rnn_output)

            if layer_idx == self.layer_num-1:
                pass
            else:
                input = current_conv_output


        current_conv_rnn_output = self.models[2*2](input,state[self.layer_num-1-2])


        output = current_conv_rnn_output.clone()
        output = self.models[2*2+1](output)
        for out_layer_idx in range(len(self.out_models)):
            output = self.out_models[out_layer_idx](output)


        warp_input_seqs = current_conv_rnn_output.clone()
        warp_input_seqs = self.models[2*2+1](warp_input_seqs)
        warp_outputs = []
        for out_layer_idx in range(10):
            warp_features.append(self.flow_model(warp_input_seqs[:, out_layer_idx]))
        warp_features = torch.stack(warp_features, 1)
        warp_outputs = self.get_warped_images(warp_features, last_frame)
        warp_outputs = torch.stack(warp_outputs,1)  


        reg_output = current_conv_rnn_output.clone()
        for out_layer_idx in range(len(self.reg_models)):
            reg_output = self.reg_models[out_layer_idx](reg_output)
        return output, warp_outputs, reg_output, current_conv_rnn_output
    
class Encode_Decode_ConvGRU(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 info,
                 ):
        super(Encode_Decode_ConvGRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.info = info
        self.models = nn.ModuleList([self.encoder,self.decoder])

    def forward(self, input):
        in_decode_frame_dat = Variable(torch.zeros(
            self.info['TRAIN']['BATCH_SIZE'],
            self.info['DATA']['OUTPUT_SEQ_LEN'],
            self.info['MODEL_NETS']['ENCODE_CELLS'][-1][1],
            self.info['MODEL_NETS']['DESHAPE'][0],
            self.info['MODEL_NETS']['DESHAPE'][0],
        ).cuda())
        encode_states = self.models[0](input)
        output, warp_output, reg_output, current_conv_output = self.models[1](in_decode_frame_dat, input, encode_states)
        return output, warp_output, reg_output, current_conv_output

class Encode_Decode_ST_ConvLSTM(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 info,
                 ):
        super(Encode_Decode_ST_ConvLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.info = info
        self.models = nn.ModuleList([self.encoder,self.decoder])

    def forward(self, input):
        # 4 10 16 13 13 
        in_decode_frame_dat = Variable(torch.zeros(
            self.info['TRAIN']['BATCH_SIZE'],
            self.info['DATA']['OUTPUT_SEQ_LEN'],
            self.info['MODEL_NETS']['ENCODE_CELLS'][-1][1],
            self.info['MODEL_NETS']['DESHAPE'][0],
            self.info['MODEL_NETS']['DESHAPE'][0],
        ).cuda())
        encode_states, M = self.models[0](input)
        output, warp_output, reg_output, hidden = self.models[1](in_decode_frame_dat, input, encode_states, M)
        return output, warp_output, reg_output, hidden

class Encoder_ST_ConvLSTM(nn.Module):
    def __init__(
            self,
            encode_shape,
            conv_rnn_cells,
            conv_cells,
            conv_m_cells,
            conv_reset_m_cells,
            info,
                 ):
        super(Encoder_ST_ConvLSTM, self).__init__()
        self.encode_shape = encode_shape
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.conv_m_cells = conv_m_cells
        self.conv_reset_m_cells = conv_reset_m_cells
        self.info = info
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        self.m_downsample = []
        self.reset_m = []

        for idx in range(len(self.conv_m_cells)):
            self.m_downsample.append(
                Conv2D(
                    cell_param=self.conv_m_cells[idx]
                ).cuda()
            )
            self.reset_m.append(
                DeConv2D(
                    cell_param=self.conv_reset_m_cells[idx]
                ).cuda()
            )
        self.m_downsample = nn.ModuleList(self.m_downsample)
        self.reset_m = nn.ModuleList(self.reset_m)

        for idx in range(self.layer_num):

            self.models.append(
                Conv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )

            self.models.append(
                SpatioTemporalLSTMCell(
                    in_channel=self.conv_rnn_cells[idx]['input_channels'],
                    num_hidden=self.conv_rnn_cells[idx]['output_channels'],
                    width=self.encode_shape[idx+1],
                    filter_size=self.conv_rnn_cells[idx]['input_to_state_kernel_size'][0],
                    stride = 1,
                    layer_norm=True,
                )
            )

        self.models = nn.ModuleList(self.models)
        self.clear_hidden_state()

    @property
    def total_layer(self):
        return self.layer_num

    def init_paramter(self,shape):
        return Variable(torch.zeros(shape).cuda(),requires_grad=True)


    def forward(self, input):

        self.n_step = input.size()[1]

        for t in range(self.n_step):
            x_t = input[:, t, :, :, :, ]
            if t == 0:
                self.cell(x_t,first_timestep=True)
            else:
                self.cell(x_t)

        return self.current_states,self.current_M

    def clear_hidden_state(self):
        self.H_history = {}
        for layer_idx in range(self.layer_num):
            self.H_history[layer_idx] = []
        self.H = {}
        self.C = {}
        self.M = {}
        self.hs = {}
        self.cs = {}
        self.ms = {}


    @property
    def history_hidden_state(self):
        return self.H_history

    @property
    def current_states(self):
        return (self.H,self.C)

    @property
    def current_M(self):
        return self.M

    def cell_reset_m(self,M):
        for idx in range(len(self.conv_m_cells)):
            M = self.reset_m[idx](M)
        return M

    def cell(self,input,first_timestep = False):

        new_states = []
        if first_timestep:
            input_shape = input.size()
            states = []
            for i in range(self.layer_num):
                h_shape = [
                    input_shape[0],
                    self.info['MODEL_NETS']['ENCODE_CELLS'][i][1],
                    self.info['MODEL_NETS']['ENSHAPE'][i+1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1]
                ]
                c_shape = [
                    input_shape[0],
                    self.info['MODEL_NETS']['ENCODE_CELLS'][i][1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1]
                ]
                m_shape = [
                    input_shape[0],
                    self.info['MODEL_NETS']['m_channels'][i],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1]
                ]


                m = self.init_paramter(m_shape)
                h = self.init_paramter(h_shape)
                c = self.init_paramter(c_shape)
                self.hs[i] = h
                self.cs[i] = c
                self.ms[i] = m

        else:
            pass
        # assert M is not None and states is not None


        for layer_idx in range(self.layer_num):

            if first_timestep == True:

                if layer_idx == 0:
                    input = self.models[layer_idx*2](input)
                    self.H[layer_idx],self.C[layer_idx],self.M[layer_idx] = self.models[layer_idx * 2 + 1](input,self.hs[layer_idx],self.cs[layer_idx], self.ms[layer_idx])

                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2 + 1](self.models[layer_idx*2](self.H[layer_idx-1]),self.hs[layer_idx],self.cs[layer_idx],self.m_downsample[layer_idx-1](self.M[layer_idx-1]))

                    continue
            else:
                if layer_idx == 0:
                    input = self.models[layer_idx * 2](input)
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2 + 1](input,
                    self.H[layer_idx], self.C[layer_idx], self.cell_reset_m(self.M[self.layer_num - 1]))

                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2 + 1](self.models[layer_idx*2](self.H[layer_idx-1]),
                    self.H[layer_idx], self.C[layer_idx],self.m_downsample[layer_idx-1](self.M[layer_idx-1]))

                    continue

        return self.H,self.C,self.M

class Decoder_ST_ConvLSTM(nn.Module):
    def __init__(
            self,
            decode_shape,
            conv_rnn_cells,
            conv_cells,
            conv_m_cells,
            conv_reset_m_cells,
            output_cells,
            reg_output_cells,
            info,
    ):
        super(Decoder_ST_ConvLSTM, self).__init__()

        self.decode_shape = decode_shape
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.conv_m_cells = conv_m_cells
        self.conv_reset_m_cells = conv_reset_m_cells
        self.output_cells = output_cells
        self.reg_output_cells = reg_output_cells
        self.info =info
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        self.m_upsample = []
        self.reset_m = []
        self.H = {}
        self.C = {}
        self.M = {}
        self.HH = {}

        for idx in range(len(self.conv_m_cells)):
            self.m_upsample.append(
                DeConv2D(
                    cell_param=self.conv_m_cells[idx]
                ).cuda()
            )
            self.reset_m.append(
                Conv2D(
                    cell_param=self.conv_reset_m_cells[idx]
                ).cuda()
            )
        self.convs = []
        self.up_convs = []
        for idx in range(self.layer_num):

            self.models.append(
                SpatioTemporalLSTMCell(
                    in_channel=self.conv_rnn_cells[idx]['input_channels'],
                    num_hidden=self.conv_rnn_cells[idx]['output_channels'],
                    width=self.decode_shape[idx],
                    filter_size=self.conv_rnn_cells[idx]['input_to_state_kernel_size'][0],
                    stride=1,
                    layer_norm=True,
                )
            )
            self.models.append(
                DeConv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )
            if idx == 0:
                self.convs.append(
                    conv_block(
                        in_ch = self.conv_cells[idx]['in_channel'],
                        out_ch = self.conv_cells[idx]['in_channel']
                    ).cuda()
                )
                self.up_convs.append(
                    up_conv(
                        in_ch=self.conv_cells[idx]['in_channel'],
                        out_ch=self.conv_cells[idx]['out_channel']
                    ).cuda()
                )
            else:
                self.convs.append(
                    conv_block(
                        in_ch=2*self.conv_cells[idx]['in_channel'],
                        out_ch=self.conv_cells[idx]['in_channel']
                    ).cuda()
                )
                self.up_convs.append(
                    up_conv(
                        in_ch=self.conv_cells[idx]['in_channel'],
                        out_ch=self.conv_cells[idx]['out_channel'],
                        output_padding=0,
                    ).cuda()
                )

        self.convs.append(
            conv_block(
                        in_ch=self.conv_cells[-1]['out_channel'],
                        out_ch=self.conv_cells[-1]['out_channel'],
                    ).cuda()
        )
        self.models = nn.ModuleList(self.models)
        self.convs = nn.ModuleList(self.convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.out_models = []
        for output_cell in self.output_cells:
            self.out_models.append(
                conv_seq.Conv2D(
                    cell_param=output_cell
                ).cuda()
            )
        self.out_models = nn.ModuleList(self.out_models)
        self.reg_models = []
        
        for reg_output_cell in reg_output_cells:
            self.reg_models.append(
                conv_seq.DeConv2D(
                    cell_param=self.conv_cells[2]
                ).cuda()
            )
            self.reg_models.append(
                conv_seq.Conv2D(
                    cell_param=reg_output_cell
                ).cuda()
            )
        self.reg_models = nn.ModuleList(self.reg_models)
        self.flow_model = nn.Conv2d(self.conv_cells[-1]['out_channel'], 2, kernel_size=1, stride=1, padding=0, bias=True)

        self.out_models = nn.ModuleList(self.out_models)
        self.seg_u_net = UNet(64, n_classes=4)
        
    def get_warped_images(self, flows, last_frame):
        wrap_preds = []
        for j in range(len(flows[0])):
            if j == 0:
                h = last_frame
            flow = flows[:, j, 0:2]
            wrap_pred = self.wrap(h, -flow)
            wrap_preds.append(wrap_pred)
            h = wrap_pred
        return wrap_preds

    def wrap(self, input, flow):
        B, C, H, W = input.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1).cuda()
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W).cuda()
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().cuda()
        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = torch.nn.functional.grid_sample(input, vgrid)
        return output

    def forward(self,input,last_frames, states,M):
        last_frame = last_frames[:,-1]
        self.n_step = input.size()[1]
        output = []
        warp_outputs = []
        warp_features = []
        current_conv_output = []
        for t in range(self.n_step):
            x_t = input[:, t, :, :, :, ]
            if t == 0:
                for layer_idx in range(self.layer_num):
                    self.H[self.layer_num-1-layer_idx] = states[0][layer_idx]
                    self.C[self.layer_num-1-layer_idx] = states[1][layer_idx]
                    self.M[self.layer_num-1-layer_idx] = M[layer_idx]
                    
                hidden = self.cell(x_t, True)
            else:
                hidden = self.cell(x_t, False)
            current_conv_output.append(hidden)
        
        current_conv_output = torch.stack(current_conv_output, 1)
        out_input = current_conv_output.clone()
        for out_layer_idx in range(10):
            output.append(self.models[self.layer_num * 2-1](out_input[:, out_layer_idx]))
        output = torch.stack(output, 1)

        for out_layer_idx in range(len(self.out_models)):
            output = self.out_models[out_layer_idx](output)
        
        warp_input_seqs = current_conv_output.clone()
        for out_layer_idx in range(10):
            warp_features.append(self.flow_model(self.models[self.layer_num * 2-1](warp_input_seqs[:, out_layer_idx])))
        warp_features = torch.stack(warp_features, 1)
        warp_outputs = self.get_warped_images(warp_features, last_frame)
        warp_outputs = torch.stack(warp_outputs,1)  

        reg_output = current_conv_output.clone()
        for out_layer_idx in range(len(self.reg_models)):
            reg_output = self.reg_models[out_layer_idx](reg_output)
        return output, warp_outputs, reg_output, current_conv_output



    @property
    def current_HH(self):
        return self.HH

    @property
    def current_states(self):
        return (self.H, self.C)

    @property
    def current_M(self):
        return self.M

    def cell_reset_m(self,M):
        for idx in range(len(self.conv_m_cells)):
            M = self.reset_m[idx](M)
        return M

    def cell(self,input,first_timestep = False):
        if first_timestep:
            for layer_idx in range(self.layer_num):
                if layer_idx == 0:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](input,self.H[layer_idx],self.C[layer_idx],self.M[layer_idx])
                else:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](self.models[2*layer_idx-1](self.H[layer_idx-1]),self.H[layer_idx],self.C[layer_idx], self.m_upsample[layer_idx-1](self.M[layer_idx-1]))

        else:
            for layer_idx in range(self.layer_num):
                if layer_idx == 0:
                    # input = self.models[layer_idx * 2](input)
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](input,
                    self.H[layer_idx], self.C[layer_idx], self.cell_reset_m(self.M[self.layer_num - 1]))

                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](self.models[layer_idx*2-1](self.H[layer_idx-1]),
                    self.H[layer_idx], self.C[layer_idx],self.m_upsample[layer_idx-1](self.M[layer_idx-1]))

                    continue
                
        
        return self.H[self.layer_num-1]
     
class Encode_Decode_CausalLSTM(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 info,
                 ):
        super(Encode_Decode_CausalLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.info = info
        self.models = nn.ModuleList([self.encoder,self.decoder])

    def forward(self, input):
        # 4 10 16 13 13 
        in_decode_frame_dat = Variable(torch.zeros(
            self.info['TRAIN']['BATCH_SIZE'],
            self.info['DATA']['OUTPUT_SEQ_LEN'],
            self.info['MODEL_NETS']['ENCODE_CELLS'][-1][1],
            self.info['MODEL_NETS']['DESHAPE'][0],
            self.info['MODEL_NETS']['DESHAPE'][0],
        ).cuda())
        encode_states, M = self.models[0](input)
        output, warp_output, reg_output, hidden = self.models[1](in_decode_frame_dat, input, encode_states, M)
        return output, warp_output, reg_output, hidden

class Encoder_CausalLSTM(nn.Module):
    def __init__(
            self,
            encode_shape,
            conv_rnn_cells,
            conv_cells,
            conv_m_cells,
            conv_reset_m_cells,
            info,
                 ):
        super(Encoder_CausalLSTM, self).__init__()
        self.encode_shape = encode_shape
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.conv_m_cells = conv_m_cells
        self.conv_reset_m_cells = conv_reset_m_cells
        self.info = info
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        self.m_downsample = []
        self.reset_m = []

        for idx in range(len(self.conv_m_cells)):
            self.m_downsample.append(
                Conv2D(
                    cell_param=self.conv_m_cells[idx]
                ).cuda()
            )
            self.reset_m.append(
                DeConv2D(
                    cell_param=self.conv_reset_m_cells[idx]
                ).cuda()
            )
        self.m_downsample = nn.ModuleList(self.m_downsample)
        self.reset_m = nn.ModuleList(self.reset_m)
        self.gradient_highway = GHU(
            64, 64, 51, 3, 1
        )
        for idx in range(self.layer_num):

            self.models.append(
                Conv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )

            self.models.append(
                CausalLSTMCell(
                    in_channel=self.conv_rnn_cells[idx]['input_channels'],
                    num_hidden=self.conv_rnn_cells[idx]['output_channels'],
                    width=self.encode_shape[idx+1],
                    filter_size=self.conv_rnn_cells[idx]['input_to_state_kernel_size'][0],
                    stride = 1,
                    layer_norm=True,
                )
            )

        self.models = nn.ModuleList(self.models)
        self.clear_hidden_state()

    @property
    def total_layer(self):
        return self.layer_num

    def init_paramter(self,shape):
        return Variable(torch.zeros(shape).cuda(),requires_grad=True)


    def forward(self, input):

        self.n_step = input.size()[1]

        for t in range(self.n_step):
            x_t = input[:, t, :, :, :, ]
            if t == 0:
                self.cell(x_t,first_timestep=True)
            else:
                self.cell(x_t)

        return self.current_states,self.current_M

    def clear_hidden_state(self):
        self.H_history = {}
        for layer_idx in range(self.layer_num):
            self.H_history[layer_idx] = []
        self.H = {}
        self.C = {}
        self.M = {}
        self.hs = {}
        self.cs = {}
        self.ms = {}


    @property
    def history_hidden_state(self):
        return self.H_history

    @property
    def current_states(self):
        return (self.H,self.C)

    @property
    def current_M(self):
        return self.M

    def cell_reset_m(self,M):
        for idx in range(len(self.conv_m_cells)):
            M = self.reset_m[idx](M)
        return M

    def cell(self,input,first_timestep = False):
        z_t = torch.zeros([4, 64, 51, 51]).cuda() # 4 64 51 51
        new_states = []
        if first_timestep:
            input_shape = input.size()
            states = []
            for i in range(self.layer_num):
                h_shape = [
                    input_shape[0],
                    self.info['MODEL_NETS']['ENCODE_CELLS'][i][1],
                    self.info['MODEL_NETS']['ENSHAPE'][i+1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1]
                ]
                c_shape = [
                    input_shape[0],
                    self.info['MODEL_NETS']['ENCODE_CELLS'][i][1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1]
                ]
                m_shape = [
                    input_shape[0],
                    self.info['MODEL_NETS']['m_channels'][i],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1]
                ]


                m = self.init_paramter(m_shape)
                h = self.init_paramter(h_shape)
                c = self.init_paramter(c_shape)
                self.hs[i] = h
                self.cs[i] = c
                self.ms[i] = m

        else:
            pass
        # assert M is not None and states is not None


        for layer_idx in range(self.layer_num):

            # if first_timestep == True:
    
            #     if layer_idx == 0:
            #         input = self.models[layer_idx*2](input)
            #         self.H[layer_idx],self.C[layer_idx],self.M[layer_idx] = self.models[layer_idx * 2 + 1](input,self.hs[layer_idx],self.cs[layer_idx], self.ms[layer_idx])
            #         z_t = self.gradient_highway(self.H[layer_idx], z_t)
            #         continue
            #     elif layer_idx == 1:
            #         self.H[layer_idx],self.C[layer_idx],self.M[layer_idx] = self.models[layer_idx * 2 + 1](self.models[layer_idx*2](z_t),self.hs[layer_idx],self.cs[layer_idx], self.ms[layer_idx])
            #         continue
            #     else:
            #         self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2 + 1](self.models[layer_idx*2](self.H[layer_idx-1]),self.hs[layer_idx],self.cs[layer_idx],self.m_downsample[layer_idx-1](self.M[layer_idx-1]))

            #         continue
            # else:
            #     if layer_idx == 0:
            #         input = self.models[layer_idx * 2](input)
            #         self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2 + 1](input, self.H[layer_idx], self.C[layer_idx], self.cell_reset_m(self.M[self.layer_num - 1]))
                    
            #         z_t = self.gradient_highway(self.H[layer_idx], z_t)
            #         continue
            #     elif layer_idx == 1:
            #         self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2 + 1](self.models[layer_idx*2](z_t), self.H[layer_idx], self.C[layer_idx],self.m_downsample[layer_idx-1](self.M[layer_idx-1]))
            #         continue
            #     else:
            #         self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2 + 1](self.models[layer_idx*2](self.H[layer_idx-1]),
            #         self.H[layer_idx], self.C[layer_idx],self.m_downsample[layer_idx-1](self.M[layer_idx-1]))

            #         continue
            if first_timestep == True:

                if layer_idx == 0:
                    input = self.models[layer_idx*2](input)
                    self.H[layer_idx],self.C[layer_idx],self.M[layer_idx] = self.models[layer_idx * 2 + 1](input,self.hs[layer_idx],self.cs[layer_idx], self.ms[layer_idx])

                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2 + 1](self.models[layer_idx*2](self.H[layer_idx-1]),self.hs[layer_idx],self.cs[layer_idx],self.m_downsample[layer_idx-1](self.M[layer_idx-1]))

                    continue
            else:
                if layer_idx == 0:
                    input = self.models[layer_idx * 2](input)
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2 + 1](input,
                    self.H[layer_idx], self.C[layer_idx], self.cell_reset_m(self.M[self.layer_num - 1]))

                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2 + 1](self.models[layer_idx*2](self.H[layer_idx-1]),
                    self.H[layer_idx], self.C[layer_idx],self.m_downsample[layer_idx-1](self.M[layer_idx-1]))

                    continue
        return self.H,self.C,self.M

class Decoder_CausalLSTM(nn.Module):
    def __init__(
            self,
            decode_shape,
            conv_rnn_cells,
            conv_cells,
            conv_m_cells,
            conv_reset_m_cells,
            output_cells,
            reg_output_cells,
            info,
    ):
        super(Decoder_CausalLSTM, self).__init__()

        self.decode_shape = decode_shape
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.conv_m_cells = conv_m_cells
        self.conv_reset_m_cells = conv_reset_m_cells
        self.output_cells = output_cells
        self.reg_output_cells = reg_output_cells
        self.info =info
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        self.m_upsample = []
        self.reset_m = []
        self.H = {}
        self.C = {}
        self.M = {}
        self.HH = {}

        for idx in range(len(self.conv_m_cells)):
            self.m_upsample.append(
                DeConv2D(
                    cell_param=self.conv_m_cells[idx]
                ).cuda()
            )
            self.reset_m.append(
                Conv2D(
                    cell_param=self.conv_reset_m_cells[idx]
                ).cuda()
            )
        self.convs = []
        self.up_convs = []
        for idx in range(self.layer_num):

            self.models.append(
                CausalLSTMCell(
                    in_channel=self.conv_rnn_cells[idx]['input_channels'],
                    num_hidden=self.conv_rnn_cells[idx]['output_channels'],
                    width=self.decode_shape[idx],
                    filter_size=self.conv_rnn_cells[idx]['input_to_state_kernel_size'][0],
                    stride=1,
                    layer_norm=True,
                )
            )
            self.models.append(
                DeConv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )
            if idx == 0:
                self.convs.append(
                    conv_block(
                        in_ch = self.conv_cells[idx]['in_channel'],
                        out_ch = self.conv_cells[idx]['in_channel']
                    ).cuda()
                )
                self.up_convs.append(
                    up_conv(
                        in_ch=self.conv_cells[idx]['in_channel'],
                        out_ch=self.conv_cells[idx]['out_channel']
                    ).cuda()
                )
            else:
                self.convs.append(
                    conv_block(
                        in_ch=2*self.conv_cells[idx]['in_channel'],
                        out_ch=self.conv_cells[idx]['in_channel']
                    ).cuda()
                )
                self.up_convs.append(
                    up_conv(
                        in_ch=self.conv_cells[idx]['in_channel'],
                        out_ch=self.conv_cells[idx]['out_channel'],
                        output_padding=0,
                    ).cuda()
                )

        self.convs.append(
            conv_block(
                        in_ch=self.conv_cells[-1]['out_channel'],
                        out_ch=self.conv_cells[-1]['out_channel'],
                    ).cuda()
        )
        self.models = nn.ModuleList(self.models)
        self.convs = nn.ModuleList(self.convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.out_models = []
        for output_cell in self.output_cells:
            self.out_models.append(
                conv_seq.Conv2D(
                    cell_param=output_cell
                ).cuda()
            )
        self.out_models = nn.ModuleList(self.out_models)
        self.reg_models = []
        
        for reg_output_cell in reg_output_cells:
            self.reg_models.append(
                conv_seq.DeConv2D(
                    cell_param=self.conv_cells[2]
                ).cuda()
            )
            self.reg_models.append(
                conv_seq.Conv2D(
                    cell_param=reg_output_cell
                ).cuda()
            )
        self.reg_models = nn.ModuleList(self.reg_models)
        self.flow_model = nn.Conv2d(self.conv_cells[-1]['out_channel'], 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_u_net = UNet(16, n_classes=4)
        self.gradient_highway = GHU(
            16, 16, 13, 3, 1
        )
    def get_warped_images(self, flows, last_frame):
        wrap_preds = []
        for j in range(len(flows[0])):
            if j == 0:
                h = last_frame
            flow = flows[:, j, 0:2]
            wrap_pred = self.wrap(h, -flow)
            wrap_preds.append(wrap_pred)
            h = wrap_pred
        return wrap_preds

    def wrap(self, input, flow):
        B, C, H, W = input.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1).cuda()
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W).cuda()
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().cuda()
        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = torch.nn.functional.grid_sample(input, vgrid)
        return output

    def forward(self,input,last_frames, states,M):
        last_frame = last_frames[:,-1]
        self.n_step = input.size()[1]
        output = []
        warp_outputs = []
        warp_features = []
        current_conv_output = []
        for t in range(self.n_step):
            x_t = input[:, t, :, :, :, ]
            if t == 0:
                for layer_idx in range(self.layer_num):
                    self.H[self.layer_num-1-layer_idx] = states[0][layer_idx]
                    self.C[self.layer_num-1-layer_idx] = states[1][layer_idx]
                    self.M[self.layer_num-1-layer_idx] = M[layer_idx]
                    
                hidden = self.cell(x_t, True)
            else:
                hidden = self.cell(x_t, False)
            current_conv_output.append(hidden)
        
        current_conv_output = torch.stack(current_conv_output, 1)
        out_input = current_conv_output.clone()
        for out_layer_idx in range(10):
            output.append(self.models[self.layer_num * 2-1](out_input[:, out_layer_idx]))
        output = torch.stack(output, 1)

        for out_layer_idx in range(len(self.out_models)):
            output = self.out_models[out_layer_idx](output)
        
        warp_input_seqs = current_conv_output.clone()
        for out_layer_idx in range(10):
            warp_features.append(self.flow_model(self.models[self.layer_num * 2-1](warp_input_seqs[:, out_layer_idx])))
        warp_features = torch.stack(warp_features, 1)
        warp_outputs = self.get_warped_images(warp_features, last_frame)
        warp_outputs = torch.stack(warp_outputs,1)  

        reg_output = current_conv_output.clone()
        for out_layer_idx in range(len(self.reg_models)):
            reg_output = self.reg_models[out_layer_idx](reg_output)
        return output, warp_outputs, reg_output, current_conv_output



    @property
    def current_HH(self):
        return self.HH

    @property
    def current_states(self):
        return (self.H, self.C)

    @property
    def current_M(self):
        return self.M

    def cell_reset_m(self,M):
        for idx in range(len(self.conv_m_cells)):
            M = self.reset_m[idx](M)
        return M

    def cell(self,input,first_timestep = False):
        z_t = torch.zeros([4, 16, 13, 13]).cuda() 
        # if first_timestep:
        #     for layer_idx in range(self.layer_num):
                
        #         if layer_idx == 0:
        #             self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](input,self.H[layer_idx],self.C[layer_idx],self.M[layer_idx])
        #             z_t = self.gradient_highway(self.H[layer_idx], z_t)
        #         elif layer_idx == 1:
        #             self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](self.models[2*layer_idx-1](z_t),self.H[layer_idx],self.C[layer_idx],self.m_upsample[layer_idx-1](self.M[layer_idx-1]))
        #         else:
        #             self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](self.models[2*layer_idx-1](self.H[layer_idx-1]),self.H[layer_idx],self.C[layer_idx],self.m_upsample[layer_idx-1](self.M[layer_idx-1]))

        # else:
        #     for layer_idx in range(self.layer_num):
        #         if layer_idx == 0:
        #             # input = self.models[layer_idx * 2](input)
        #             self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](input,
        #             self.H[layer_idx], self.C[layer_idx], self.cell_reset_m(self.M[self.layer_num - 1]))
        #             z_t = self.gradient_highway(self.H[layer_idx], z_t)
        #             continue
        #         elif layer_idx == 1:
        #             self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](self.models[layer_idx*2-1](z_t),
        #             self.H[layer_idx], self.C[layer_idx],self.m_upsample[layer_idx-1](self.M[layer_idx-1]))
        #             continue
        #         else:
        #             self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](self.models[layer_idx*2-1](self.H[layer_idx-1]),
        #             self.H[layer_idx], self.C[layer_idx],self.m_upsample[layer_idx-1](self.M[layer_idx-1]))
        #             continue
                
        if first_timestep:
            for layer_idx in range(self.layer_num):
                if layer_idx == 0:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](input,self.H[layer_idx],self.C[layer_idx],self.M[layer_idx])
                else:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](self.models[2*layer_idx-1](self.H[layer_idx-1]),self.H[layer_idx],self.C[layer_idx], self.m_upsample[layer_idx-1](self.M[layer_idx-1]))

        else:
            for layer_idx in range(self.layer_num):
                if layer_idx == 0:
                    # input = self.models[layer_idx * 2](input)
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](input,
                    self.H[layer_idx], self.C[layer_idx], self.cell_reset_m(self.M[self.layer_num - 1]))

                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](self.models[layer_idx*2-1](self.H[layer_idx-1]),
                    self.H[layer_idx], self.C[layer_idx],self.m_upsample[layer_idx-1](self.M[layer_idx-1]))

                    continue
        return self.H[self.layer_num-1]

class Encode_Decode_SAConvLSTM(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 info,
                 ):
        super(Encode_Decode_SAConvLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.info = info
        self.models = nn.ModuleList([self.encoder,self.decoder])

    def forward(self, input):
        # 4 10 16 13 13 
        in_decode_frame_dat = Variable(torch.zeros(
            self.info['TRAIN']['BATCH_SIZE'],
            self.info['DATA']['OUTPUT_SEQ_LEN'],
            self.info['MODEL_NETS']['ENCODE_CELLS'][-1][1],
            self.info['MODEL_NETS']['DESHAPE'][0],
            self.info['MODEL_NETS']['DESHAPE'][0],
        ).cuda())
        encode_states, M = self.models[0](input)
        output, warp_output, reg_output, hidden = self.models[1](in_decode_frame_dat, input, encode_states, M)
        return output, warp_output, reg_output, hidden

class Encoder_SAConvLSTM(nn.Module):
    def __init__(
            self,
            encode_shape,
            conv_rnn_cells,
            conv_cells,
            conv_m_cells,
            conv_reset_m_cells,
            info,
                 ):
        super(Encoder_SAConvLSTM, self).__init__()
        self.encode_shape = encode_shape
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.conv_m_cells = conv_m_cells
        self.conv_reset_m_cells = conv_reset_m_cells
        self.info = info
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        self.m_downsample = []
        self.reset_m = []

        for idx in range(len(self.conv_m_cells)):
            self.m_downsample.append(
                Conv2D(
                    cell_param=self.conv_m_cells[idx]
                ).cuda()
            )
            self.reset_m.append(
                DeConv2D(
                    cell_param=self.conv_reset_m_cells[idx]
                ).cuda()
            )
        self.m_downsample = nn.ModuleList(self.m_downsample)
        self.reset_m = nn.ModuleList(self.reset_m)

        for idx in range(self.layer_num):

            self.models.append(
                Conv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )

            self.models.append(
                SAConvLSTMCell(
                    in_channel=self.conv_rnn_cells[idx]['input_channels'],
                    num_hidden=self.conv_rnn_cells[idx]['output_channels'],
                    width=self.encode_shape[idx+1],
                    filter_size=self.conv_rnn_cells[idx]['input_to_state_kernel_size'][0],
                    stride = 1,
                    layer_norm=True,
                )
            )

        self.models = nn.ModuleList(self.models)
        self.clear_hidden_state()

    @property
    def total_layer(self):
        return self.layer_num

    def init_paramter(self,shape):
        return Variable(torch.zeros(shape).cuda(),requires_grad=True)


    def forward(self, input):

        self.n_step = input.size()[1]

        for t in range(self.n_step):
            x_t = input[:, t, :, :, :, ]
            if t == 0:
                self.cell(x_t,first_timestep=True)
            else:
                self.cell(x_t)

        return self.current_states,self.current_M

    def clear_hidden_state(self):
        self.H_history = {}
        for layer_idx in range(self.layer_num):
            self.H_history[layer_idx] = []
        self.H = {}
        self.C = {}
        self.M = {}
        self.hs = {}
        self.cs = {}
        self.ms = {}


    @property
    def history_hidden_state(self):
        return self.H_history

    @property
    def current_states(self):
        return (self.H,self.C)

    @property
    def current_M(self):
        return self.M

    def cell_reset_m(self,M):
        for idx in range(len(self.conv_m_cells)):
            M = self.reset_m[idx](M)
        return M

    def cell(self,input,first_timestep = False):

        new_states = []
        if first_timestep:
            input_shape = input.size()
            states = []
            for i in range(self.layer_num):
                h_shape = [
                    input_shape[0],
                    self.info['MODEL_NETS']['ENCODE_CELLS'][i][1],
                    self.info['MODEL_NETS']['ENSHAPE'][i+1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1]
                ]
                c_shape = [
                    input_shape[0],
                    self.info['MODEL_NETS']['ENCODE_CELLS'][i][1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1]
                ]
                m_shape = [
                    input_shape[0],
                    self.info['MODEL_NETS']['m_channels'][i],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1]
                ]


                m = self.init_paramter(m_shape)
                h = self.init_paramter(h_shape)
                c = self.init_paramter(c_shape)
                self.hs[i] = h
                self.cs[i] = c
                self.ms[i] = m

        else:
            pass
        # assert M is not None and states is not None


        for layer_idx in range(self.layer_num):

            if first_timestep == True:

                if layer_idx == 0:
                    input = self.models[layer_idx*2](input)
                    self.H[layer_idx],self.C[layer_idx],self.M[layer_idx] = self.models[layer_idx * 2 + 1](input,self.hs[layer_idx],self.cs[layer_idx], self.ms[layer_idx])

                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2 + 1](self.models[layer_idx*2](self.H[layer_idx-1]),self.hs[layer_idx],self.cs[layer_idx],self.m_downsample[layer_idx-1](self.M[layer_idx-1]))

                    continue
            else:
                if layer_idx == 0:
                    input = self.models[layer_idx * 2](input)
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2 + 1](input,
                    self.H[layer_idx], self.C[layer_idx], self.cell_reset_m(self.M[self.layer_num - 1]))

                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2 + 1](self.models[layer_idx*2](self.H[layer_idx-1]),
                    self.H[layer_idx], self.C[layer_idx],self.m_downsample[layer_idx-1](self.M[layer_idx-1]))

                    continue

        return self.H,self.C,self.M

class Decoder_SAConvLSTM(nn.Module):
    def __init__(
            self,
            decode_shape,
            conv_rnn_cells,
            conv_cells,
            conv_m_cells,
            conv_reset_m_cells,
            output_cells,
            reg_output_cells,
            info,
    ):
        super(Decoder_SAConvLSTM, self).__init__()

        self.decode_shape = decode_shape
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.conv_m_cells = conv_m_cells
        self.conv_reset_m_cells = conv_reset_m_cells
        self.output_cells = output_cells
        self.reg_output_cells = reg_output_cells
        self.info =info
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        self.m_upsample = []
        self.reset_m = []
        self.H = {}
        self.C = {}
        self.M = {}
        self.HH = {}

        for idx in range(len(self.conv_m_cells)):
            self.m_upsample.append(
                DeConv2D(
                    cell_param=self.conv_m_cells[idx]
                ).cuda()
            )
            self.reset_m.append(
                Conv2D(
                    cell_param=self.conv_reset_m_cells[idx]
                ).cuda()
            )
        self.convs = []
        self.up_convs = []
        for idx in range(self.layer_num):

            self.models.append(
                SAConvLSTMCell(
                    in_channel=self.conv_rnn_cells[idx]['input_channels'],
                    num_hidden=self.conv_rnn_cells[idx]['output_channels'],
                    width=self.decode_shape[idx],
                    filter_size=self.conv_rnn_cells[idx]['input_to_state_kernel_size'][0],
                    stride=1,
                    layer_norm=True,
                )
            )
            self.models.append(
                DeConv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )
            if idx == 0:
                self.convs.append(
                    conv_block(
                        in_ch = self.conv_cells[idx]['in_channel'],
                        out_ch = self.conv_cells[idx]['in_channel']
                    ).cuda()
                )
                self.up_convs.append(
                    up_conv(
                        in_ch=self.conv_cells[idx]['in_channel'],
                        out_ch=self.conv_cells[idx]['out_channel']
                    ).cuda()
                )
            else:
                self.convs.append(
                    conv_block(
                        in_ch=2*self.conv_cells[idx]['in_channel'],
                        out_ch=self.conv_cells[idx]['in_channel']
                    ).cuda()
                )
                self.up_convs.append(
                    up_conv(
                        in_ch=self.conv_cells[idx]['in_channel'],
                        out_ch=self.conv_cells[idx]['out_channel'],
                        output_padding=0,
                    ).cuda()
                )

        self.convs.append(
            conv_block(
                        in_ch=self.conv_cells[-1]['out_channel'],
                        out_ch=self.conv_cells[-1]['out_channel'],
                    ).cuda()
        )
        self.models = nn.ModuleList(self.models)
        self.convs = nn.ModuleList(self.convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.out_models = []
        for output_cell in self.output_cells:
            self.out_models.append(
                conv_seq.Conv2D(
                    cell_param=output_cell
                ).cuda()
            )
        self.out_models = nn.ModuleList(self.out_models)
        self.reg_models = []
        
        for reg_output_cell in reg_output_cells:
            self.reg_models.append(
                conv_seq.DeConv2D(
                    cell_param=self.conv_cells[2]
                ).cuda()
            )
            self.reg_models.append(
                conv_seq.Conv2D(
                    cell_param=reg_output_cell
                ).cuda()
            )
        self.reg_models = nn.ModuleList(self.reg_models)
        self.flow_model = nn.Conv2d(self.conv_cells[-1]['out_channel'], 4, kernel_size=1, stride=1, padding=0, bias=True)

        self.out_models = nn.ModuleList(self.out_models)
        self.seg_u_net = UNet(64, n_classes=4)
        
    def get_warped_images(self, flows, last_frame):
        wrap_preds = []
        for j in range(len(flows[0])):
            if j == 0:
                h = last_frame
            flow = flows[:, j, 0:2]
            wrap_pred = self.wrap(h, -flow)
            wrap_preds.append(wrap_pred)
            h = wrap_pred
        return wrap_preds
    
    def forward(self,input,last_frames, states,M):
        last_frame = last_frames[:,-1]
        self.n_step = input.size()[1]
        output = []
        warp_outputs = []
        warp_features = []
        warp_outputs = []
        current_conv_output = []
        for t in range(self.n_step):
            x_t = input[:, t, :, :, :, ]
            if t == 0:
                for layer_idx in range(self.layer_num):
                    self.H[self.layer_num-1-layer_idx] = states[0][layer_idx]
                    self.C[self.layer_num-1-layer_idx] = states[1][layer_idx]
                    self.M[self.layer_num-1-layer_idx] = M[layer_idx]
                    
                hidden = self.cell(x_t, True)
            else:
                hidden = self.cell(x_t, False)
            current_conv_output.append(hidden)
        
        current_conv_output = torch.stack(current_conv_output, 1)
        out_input = current_conv_output.clone()
        for out_layer_idx in range(10):
            output.append(self.models[self.layer_num * 2-1](out_input[:, out_layer_idx]))
        output = torch.stack(output, 1)

        for out_layer_idx in range(len(self.out_models)):
            output = self.out_models[out_layer_idx](output)
        
        warp_input_seqs = current_conv_output.clone()
        for out_layer_idx in range(10):
            warp_features.append(self.models[self.layer_num * 2-1](warp_input_seqs[:, out_layer_idx]))
        warp_features = torch.stack(warp_features, 1)
        print(warp_features.shape)
        for t in range(warp_features.shape[1]):
            warp_output = self.seg_u_net(warp_features[:,t])
            warp_outputs.append(warp_output)
        warp_outputs = torch.stack(warp_outputs,1)  

        reg_output = current_conv_output.clone()
        for out_layer_idx in range(len(self.reg_models)):
            reg_output = self.reg_models[out_layer_idx](reg_output)
        return output, warp_outputs, reg_output, current_conv_output


    @property
    def current_HH(self):
        return self.HH

    @property
    def current_states(self):
        return (self.H, self.C)

    @property
    def current_M(self):
        return self.M

    def cell_reset_m(self,M):
        for idx in range(len(self.conv_m_cells)):
            M = self.reset_m[idx](M)
        return M

    def cell(self,input,first_timestep = False):
        if first_timestep:
            for layer_idx in range(self.layer_num):
                if layer_idx == 0:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](input,self.H[layer_idx],self.C[layer_idx],self.M[layer_idx])
                else:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](self.models[2*layer_idx-1](self.H[layer_idx-1]),self.H[layer_idx],self.C[layer_idx], self.m_upsample[layer_idx-1](self.M[layer_idx-1]))

        else:
            for layer_idx in range(self.layer_num):
                if layer_idx == 0:
                    # input = self.models[layer_idx * 2](input)
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](input,
                    self.H[layer_idx], self.C[layer_idx], self.cell_reset_m(self.M[self.layer_num - 1]))

                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx], self.M[layer_idx] = self.models[layer_idx * 2](self.models[layer_idx*2-1](self.H[layer_idx-1]),
                    self.H[layer_idx], self.C[layer_idx],self.m_upsample[layer_idx-1](self.M[layer_idx-1]))

                    continue
                
        
        return self.H[self.layer_num-1]

class Encode_Decode_CMSLSTM(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 info,
                 ):
        super(Encode_Decode_CMSLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.info = info
        self.models = nn.ModuleList([self.encoder,self.decoder])

    def forward(self, input):
        # 4 10 16 13 13 
        in_decode_frame_dat = Variable(torch.zeros(
            self.info['TRAIN']['BATCH_SIZE'],
            self.info['DATA']['OUTPUT_SEQ_LEN'],
            self.info['MODEL_NETS']['ENCODE_CELLS'][-1][1],
            self.info['MODEL_NETS']['DESHAPE'][0],
            self.info['MODEL_NETS']['DESHAPE'][0],
        ).cuda())
        encode_states = self.models[0](input)
        output, seg_out, warp_out, hidden = self.models[1](in_decode_frame_dat, input, encode_states)
        return output, seg_out, warp_out , hidden

class Encode_CMSLSTM(nn.Module):
    def __init__(
            self,
            encode_shape,
            conv_rnn_cells,
            conv_cells,
            info,
            ):
        super(Encode_CMSLSTM, self).__init__()
        self.encode_shape = encode_shape
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.info = info
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        for idx in range(self.layer_num):

            self.models.append(
                Conv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )

            self.models.append(
                CMSLSTM_cell(
                    in_channel=self.conv_rnn_cells[idx]['input_channels'],
                    num_hidden=self.conv_rnn_cells[idx]['output_channels'],
                    width=self.encode_shape[idx+1],
                    filter_size=self.conv_rnn_cells[idx]['input_to_state_kernel_size'][0],
                    stride = 1,
                    layer_norm=True,
                )
            )

        self.models = nn.ModuleList(self.models)
        self.clear_hidden_state()
    @property
    def total_layer(self):
        return self.layer_num

    def init_paramter(self,shape):
        return Variable(torch.zeros(shape).cuda(),requires_grad=True)
    def forward(self, input):

        self.n_step = input.size()[1]

        for t in range(self.n_step):
            x_t = input[:, t, :, :, :, ]
            if t == 0:
                self.cell(x_t,first_timestep=True)
            else:
                self.cell(x_t)

        return self.current_states
    def clear_hidden_state(self):
        self.H_history = {}
        for layer_idx in range(self.layer_num):
            self.H_history[layer_idx] = []
        self.H = {}
        self.C = {}
        self.hs = {}
        self.cs = {}
    @property
    def history_hidden_state(self):
        return self.H_history

    @property
    def current_states(self):
        return (self.H,self.C)
    def cell(self,input,first_timestep = False):
        if first_timestep:
            input_shape = input.size()
            for i in range(self.layer_num):
                h_shape = [
                    input_shape[0],
                    self.info['MODEL_NETS']['ENCODE_CELLS'][i][1],
                    self.info['MODEL_NETS']['ENSHAPE'][i+1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1]
                ]
                c_shape = [
                    input_shape[0],
                    self.info['MODEL_NETS']['ENCODE_CELLS'][i][1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1],
                    self.info['MODEL_NETS']['ENSHAPE'][i + 1]
                ]

                h = self.init_paramter(h_shape)
                c = self.init_paramter(c_shape)
                self.hs[i] = h
                self.cs[i] = c

        else:
            pass


        for layer_idx in range(self.layer_num):

            if first_timestep == True:

                if layer_idx == 0:
                    input = self.models[layer_idx*2](input)
                    self.H[layer_idx],self.C[layer_idx] = self.models[layer_idx * 2 + 1](input,self.hs[layer_idx],self.cs[layer_idx])
                    
                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx] = self.models[layer_idx * 2 + 1](self.models[layer_idx*2](self.H[layer_idx-1]),self.hs[layer_idx],self.cs[layer_idx])

                    continue
            else:
                if layer_idx == 0:
                    input = self.models[layer_idx * 2](input)
                    self.H[layer_idx], self.C[layer_idx] = self.models[layer_idx * 2 + 1](input, self.H[layer_idx], self.C[layer_idx])

                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx] = self.models[layer_idx * 2 + 1](self.models[layer_idx*2](self.H[layer_idx-1]), self.H[layer_idx], self.C[layer_idx])

                    continue

        return self.H,self.C

class Decode_CMSLSTM(nn.Module):
    def __init__(
            self,
            decode_shape,
            conv_rnn_cells,
            conv_cells,
            output_cells,
            reg_output_cells,
            info,
    ):
        super(Decode_CMSLSTM, self).__init__()

        self.decode_shape = decode_shape
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.output_cells = output_cells
        self.reg_output_cells = reg_output_cells
        self.info =info
        self.models = []
        self.layer_num = len(self.conv_rnn_cells)
        self.H = {}
        self.C = {}
        self.M = {}
        self.HH = {}

      
        self.convs = []
        self.up_convs = []
        for idx in range(self.layer_num):

            self.models.append(
                CMSLSTM_cell(
                    in_channel=self.conv_rnn_cells[idx]['input_channels'],
                    num_hidden=self.conv_rnn_cells[idx]['output_channels'],
                    width=self.decode_shape[idx],
                    filter_size=self.conv_rnn_cells[idx]['input_to_state_kernel_size'][0],
                    stride=1,
                    layer_norm=True,
                )
            )
            self.models.append(
                DeConv2D(
                    cell_param=self.conv_cells[idx]
                ).cuda()
            )
            if idx == 0:
                self.convs.append(
                    conv_block(
                        in_ch = self.conv_cells[idx]['in_channel'],
                        out_ch = self.conv_cells[idx]['in_channel']
                    ).cuda()
                )
                self.up_convs.append(
                    up_conv(
                        in_ch=self.conv_cells[idx]['in_channel'],
                        out_ch=self.conv_cells[idx]['out_channel']
                    ).cuda()
                )
            else:
                self.convs.append(
                    conv_block(
                        in_ch=2*self.conv_cells[idx]['in_channel'],
                        out_ch=self.conv_cells[idx]['in_channel']
                    ).cuda()
                )
                self.up_convs.append(
                    up_conv(
                        in_ch=self.conv_cells[idx]['in_channel'],
                        out_ch=self.conv_cells[idx]['out_channel'],
                        output_padding=0,
                    ).cuda()
                )

        self.convs.append(
            conv_block(
                        in_ch=self.conv_cells[-1]['out_channel'],
                        out_ch=self.conv_cells[-1]['out_channel'],
                    ).cuda()
        )
        self.models = nn.ModuleList(self.models)
        self.convs = nn.ModuleList(self.convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.out_models = []
        for output_cell in self.output_cells:
            self.out_models.append(
                conv_seq.Conv2D(
                    cell_param=output_cell
                ).cuda()
            )
        self.out_models = nn.ModuleList(self.out_models)
        self.reg_models = []
        
        for reg_output_cell in reg_output_cells:
            self.reg_models.append(
                conv_seq.DeConv2D(
                    cell_param=self.conv_cells[2]
                ).cuda()
            )
            self.reg_models.append(
                conv_seq.Conv2D(
                    cell_param=reg_output_cell
                ).cuda()
            )
        self.reg_models = nn.ModuleList(self.reg_models)
        self.flow_model = nn.Conv2d(self.conv_cells[-1]['out_channel'], 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_u_net = UNet(16, n_classes=4)
        
    def get_warped_images(self, flows, last_frame):
        wrap_preds = []
        for j in range(len(flows[0])):
            if j == 0:
                h = last_frame
            flow = flows[:, j, 0:2]
            wrap_pred = self.wrap(h, -flow)
            wrap_preds.append(wrap_pred)
            h = wrap_pred
        return wrap_preds

    def wrap(self, input, flow):
        B, C, H, W = input.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1).cuda()
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W).cuda()
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().cuda()
        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = torch.nn.functional.grid_sample(input, vgrid)
        return output
        
    
    def forward(self,input,last_frames, states):
        last_frame = last_frames[:,-1]
        self.n_step = input.size()[1]
        output = []
        warp_outputs = []
        warp_features = []
        current_conv_output = []
        for t in range(self.n_step):
            x_t = input[:, t, :, :, :, ]
            if t == 0:
                for layer_idx in range(self.layer_num):
                    self.H[self.layer_num-1-layer_idx] = states[0][layer_idx]
                    self.C[self.layer_num-1-layer_idx] = states[1][layer_idx]
                    
                hidden = self.cell(x_t, True)
            else:
                hidden = self.cell(x_t, False)
            current_conv_output.append(hidden)
        
        current_conv_output = torch.stack(current_conv_output, 1)
        out_input = current_conv_output.clone()
        for out_layer_idx in range(10):
            output.append(self.models[self.layer_num * 2-1](out_input[:, out_layer_idx]))
        output = torch.stack(output, 1)

        for out_layer_idx in range(len(self.out_models)):
            output = self.out_models[out_layer_idx](output)
        
        warp_input_seqs = current_conv_output.clone()
        for out_layer_idx in range(10):
            warp_features.append(self.flow_model(self.models[self.layer_num * 2-1](warp_input_seqs[:, out_layer_idx])))
        warp_features = torch.stack(warp_features, 1)
        warp_outputs = self.get_warped_images(warp_features, last_frame)
        warp_outputs = torch.stack(warp_outputs,1) 

        reg_output = current_conv_output.clone()
        for out_layer_idx in range(len(self.reg_models)):
            reg_output = self.reg_models[out_layer_idx](reg_output)
        return output, warp_outputs, reg_output, current_conv_output

    @property
    def current_HH(self):
        return self.HH

    @property
    def current_states(self):
        return (self.H, self.C)



    def cell(self,input,first_timestep = False):
        if first_timestep:
            for layer_idx in range(self.layer_num):
                if layer_idx == 0:
                    self.H[layer_idx], self.C[layer_idx] = self.models[layer_idx * 2](input,self.H[layer_idx],self.C[layer_idx])
                else:
                    self.H[layer_idx], self.C[layer_idx] = self.models[layer_idx * 2](self.models[2*layer_idx-1](self.H[layer_idx-1]),self.H[layer_idx],self.C[layer_idx])

        else:
            for layer_idx in range(self.layer_num):
                if layer_idx == 0:
                    # input = self.models[layer_idx * 2](input)
                    self.H[layer_idx], self.C[layer_idx] = self.models[layer_idx * 2](input,
                    self.H[layer_idx], self.C[layer_idx])

                    continue
                else:
                    self.H[layer_idx], self.C[layer_idx] = self.models[layer_idx * 2](self.models[layer_idx*2-1](self.H[layer_idx-1]),
                    self.H[layer_idx], self.C[layer_idx])

                    continue
                
        
        return self.H[self.layer_num-1]


    