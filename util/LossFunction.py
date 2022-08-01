import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F
# from apex import amp



class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class MultiLoss(nn.Module):
    def __init__(self,loss_list_length):
        super(MultiLoss, self).__init__()
        self._sigmas_sq = []
        for i in range(loss_list_length):
            self._sigmas_sq.append(
                torch.nn.init.uniform_(Parameter(torch.Tensor(1).cuda()),a=0.2, b=1)
            )

    def forward(self,loss_list):
        # print(self._sigmas_sq)
        self._loss_list = loss_list
        factor = 1.0/(2.0*self._sigmas_sq[0]*self._sigmas_sq[0])
        loss = factor*self._loss_list[0] + torch.log(self._sigmas_sq[0])

        # print(factor, factor * self._loss_list[0], torch.log(self._sigmas_sq[0]))

        for i in range(1,len(self._sigmas_sq)):
            factor = 1.0/(2.0*self._sigmas_sq[i]*self._sigmas_sq[i])
            loss = loss + factor * self._loss_list[i] + torch.log(self._sigmas_sq[i])


        return loss

def loss_gradient_difference(real_image,generated): # b x c x h x w
    true_x_shifted_right = real_image[:,:,1:,:]# 32 x 3 x 255 x 256
    true_x_shifted_left = real_image[:,:,:-1,:]
    true_x_gradient = torch.abs(true_x_shifted_left - true_x_shifted_right)

    generated_x_shift_right = generated[:,:,1:,:]# 32 x 3 x 255 x 256
    generated_x_shift_left = generated[:,:,:-1,:]
    generated_x_griednt = torch.abs(generated_x_shift_left - generated_x_shift_right)

    difference_x = true_x_gradient - generated_x_griednt

    loss_x_gradient = (torch.sum(difference_x)**2)/2 # tf.nn.l2_loss(true_x_gradient - generated_x_gradient)

    true_y_shifted_right = real_image[:,:,:,1:]
    true_y_shifted_left = real_image[:,:,:,:-1]
    true_y_gradient = torch.abs(true_y_shifted_left - true_y_shifted_right)

    generated_y_shift_right = generated[:,:,:,1:]
    generated_y_shift_left = generated[:,:,:,:-1]
    generated_y_griednt = torch.abs(generated_y_shift_left - generated_y_shift_right)

    difference_y = true_y_gradient - generated_y_griednt
    loss_y_gradient = (torch.sum(difference_y)**2)/2 # tf.nn.l2_loss(true_y_gradient - generated_y_gradient)

    igdl = loss_x_gradient + loss_y_gradient
    return igdl

def get_st_warp_weights(model, hidden, target, origin_input):
    hidden.requires_grad_(True)
    warp_features = []
    warp_input_seqs = hidden.clone()
    for out_layer_idx in range(10):
        warp_features.append(model.flow_model(model.models[model.layer_num * 2-1](warp_input_seqs[:, out_layer_idx])))
    warp_features = torch.stack(warp_features, 1)
    warp_outputs = model.get_warped_images(warp_features, origin_input)
    warp_outputs = torch.stack(warp_outputs,1)  
    mse = torch.pow((warp_outputs - target), 2).mean()
    grads = torch.autograd.grad(mse, hidden, retain_graph = True)[0]
    return grads.detach()

def compute_rank_correlation(att, grad_att):
    """
    Function that measures Spearman’s correlation coefficient between target logits and output logits:
    att: [n, m]
    grad_att: [n, m]
    """
    def _rank_correlation_(att_map, att_gd):
        n = torch.tensor(att_map.shape[1])
        upper = 6 * torch.sum((att_gd - att_map).pow(2), dim=1)
        down = n * (n.pow(2) - 1.0)
        return (1.0 - (upper / down))

    att = att.sort(dim=1)[1]
    grad_att = grad_att.sort(dim=1)[1]
    
    n = torch.tensor(att.shape[1])
    x =  n * (n.pow(2) - 1.0)
    print(x)
    correlation = _rank_correlation_(att.float(), grad_att.float())
    return correlation

def get_st_reg_weights(model, hidden, target):
    hidden.requires_grad_(True)
    output = []
    for out_layer_idx in range(10):
        output.append(model.models[model.layer_num * 2-1](hidden[:, out_layer_idx]))
    output = torch.stack(output, 1)
    for out_layer_idx in range(len(model.out_models)):
        output = model.out_models[out_layer_idx](output)
    mse = torch.pow((output - target), 2).mean()
    grads = torch.autograd.grad(mse, hidden, retain_graph = True)[0]
    return grads.detach()
    
def get_st_regression_weights(model, hidden, target):
    hidden.requires_grad_(True)
    reg_output = hidden
    for out_layer_idx in range(len(model.reg_models)):
        reg_output = model.reg_models[out_layer_idx](reg_output)
    mse = torch.pow((reg_output - target), 2).mean()
    grads = torch.autograd.grad(mse, hidden, retain_graph = True)[0]
    return grads.detach()

def get_reg_weights(model, hidden, target):
    hidden.requires_grad_(True)
    output = model.models[2*2+1](hidden)
    for out_layer_idx in range(len(model.out_models)):
        output = model.out_models[out_layer_idx](output)
    mse = torch.pow((output - target), 2).mean()
    grads = torch.autograd.grad(mse, hidden, retain_graph = True)[0]
    return grads.detach()
    
def get_warp_weights(model, hidden, target, last_frame):
    hidden.requires_grad_(True)
    warp_features = []
    warp_input_seqs = model.models[2*2+1](hidden)
    warp_outputs = []
    for out_layer_idx in range(10):
        warp_features.append(model.flow_model(warp_input_seqs[:, out_layer_idx]))
    warp_features = torch.stack(warp_features, 1)
    warp_outputs = model.get_warped_images(warp_features, last_frame)
    warp_outputs = torch.stack(warp_outputs,1)
    mse = torch.pow((warp_outputs - target), 2).mean()
    grads = torch.autograd.grad(mse, hidden, retain_graph = True)[0]
    return grads.detach()

def get_regression_weights(model, hidden, target):
    hidden.requires_grad_(True)
    reg_output = hidden
    for out_layer_idx in range(len(model.reg_models)):
        reg_output = model.reg_models[out_layer_idx](reg_output)
    mse = torch.pow((reg_output - target), 2).mean()
    grads = torch.autograd.grad(mse, hidden, retain_graph = True)[0]
    return grads.detach()

if __name__ == '__main__':
    x = torch.randn(10, 64*64*64)
    y = torch.randn(10, 64*64*64)
    h = compute_rank_correlation(x, y)