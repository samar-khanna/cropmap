import numpy as np
import torch
import dtwco
from torch import nn
from tslearn.metrics import dtw_path


class DTWNET_BASE(nn.Module):
  def __init__(self, _dtwlayer_list):
    super(DTWNET_BASE, self).__init__()
    self.num_dtwlayer = len(_dtwlayer_list)
    self.dtwlayers = nn.ModuleList(_dtwlayer_list)
    self.dtwlayer_outlen = _dtwlayer_list[-1].out_len
  def construct_dtw(self, input):
    pass
  def forward(self, input):
    pass



class DTWNET(DTWNET_BASE):
  def __init__(self, _n_class, _dtwlayer_list, num_mlp_layers):
    super(DTWNET, self).__init__(_dtwlayer_list)
    self.mlps = nn.ModuleList([MLP(num_mlp_layers, self.dtwlayer_outlen, _n_class)])
    self.n_class =_n_class

  def construct_dtw(self, input):
    pass


  def forward(self, x):
    # t = self.construct_dtw(input)
    # t = t.view(input.shape[0], -1)
    # return self.mlps[0](t)

    n = len(x)
    b, c, h, w = x[0].shape
    x = torch.cat(x, dim=0)
    x = x.view(-1, n, c)  # (bhw, #timepoints, #channels)
    x = self.construct_dtw(x)
    x = x.view(x.shape[0], -1)
    x = self.mlps[0](x)
    x = x.view(b, self.n_class, h, w)
    return x

class DTWNET_SINGLE(DTWNET):
  def __init__(self, _n_class, n_filter, kernel_shape_str, feature_len, num_mlp_layers):
    kernel_shape = tuple([int(i) for i in kernel_shape_str.split()])
    dtw_list = []
    for i in range(n_filter):
        dtw_list.append(DTW_FULL_DTWCO(kernel_shape, feature_len))
    super(DTWNET_SINGLE, self).__init__(_n_class, [DTWLAYER(dtw_list)], num_mlp_layers)

  @classmethod
  def create(cls, config, num_classes):
    return cls(num_classes, **config["classifier_kwargs"])


  def construct_dtw(self, input):
    t = self.dtwlayers[0](input)
    for i in range(1,len(self.dtwlayers)):
      t = self.dtwlayers[i](t)
    t = t.view(input.shape[0], -1)
    return t

class DTW(nn.Module):
  def __init__(self, _kernel_shape, _input_len, **kwargs):
    super(DTW, self).__init__()
    self.kernel_shape = _kernel_shape
    self.input_len = _input_len
    self.kernel = nn.Parameter(torch.randn(_kernel_shape))
  def compute(self, input):
    pass
  def forward(self, input):
    return self.compute(input)




class DTW_FULL_DTWCO(DTW):
  def __init__(self, _kernel_shape, _input_len, **kwargs):
    super(DTW_FULL_DTWCO, self).__init__(_kernel_shape, _input_len, **kwargs)
    self.out_len = 1

  def compute(self, input):
    path, cost = dtw_path(self.kernel.detach().cpu().numpy(), input.detach().cpu().numpy())
    return sum([ sum((self.kernel[i] - input[j])**2) for i,j in path])
    # dist, cost, path = dtwco.dtw(self.kernel.detach().cpu().numpy(), input.detach().cpu().numpy(), metric='sqeuclidean', dist_only=False)
    #  return sum([(self.kernel[path[0][i]]-input[path[1][i]])**2 for i in range(len(path[0]))])


class DTWLAYER(nn.Module):
  def __init__(self, _dtw_list):
    super(DTWLAYER, self).__init__()
    self.num_filter = len(_dtw_list)
    self.filters = nn.ModuleList(_dtw_list)
    self.filter_outlen = _dtw_list[0].out_len
    self.out_len = self.num_filter*self.filter_outlen
  def forward_one_batch(self, input):
    out = torch.zeros((self.num_filter, self.filter_outlen))
    for i in range(self.num_filter):
      out[i] = self.filters[i].forward(input)
    return out
  def forward(self, input):
    out = torch.tensor([], device='cuda').new_empty(input.shape[0], self.num_filter, self.filter_outlen)
    for k in range(input.shape[0]): # batch size
      out[k] = self.forward_one_batch(input[k])
    return out

class MLP(nn.Module):
  def __init__(self, _n_layer, _input_len, _output_len):
    super(MLP, self).__init__()
    self.n_layer = _n_layer
    tmp = [nn.Linear(_input_len, 128)]
    for i in range(1, _n_layer-1):
      tmp.append(nn.Linear(128, 128))
    self.linears = nn.ModuleList(tmp)
    self.logits = nn.Linear(128, _output_len)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input):
    t = input.view(input.shape[0],-1)
    for i in range(self.n_layer-1):
      t = self.linears[i](t)
      t = self.relu(t)
    t = self.logits(t)
    return self.softmax(t)

