import torch
import torch.nn as nn
from tslearn.metrics import dtw


class kNN(nn.Module):
    def __init__(self, num_classes, k=5, metric="L2", use_ndvi=False):
        super().__init__()
        self.k = k
        self.dummy_param = nn.Parameter(torch.tensor([0.05]))
        self.metric = metric
        self.memory_x = None
        self.memory_y = None
        self.eye = torch.eye(num_classes)
        self.use_ndvi = use_ndvi

    @classmethod
    def create(cls, config, num_classes):
        return cls(num_classes, **config["classifier_kwargs"])

    def forward(self, x, y):
        ndvi = [((im[:, 4] - im[:, 3]) / (im[:, 4] + im[:, 3] + 1e-4)).unsqueeze(1) for im in x]
        if self.use_ndvi: x = ndvi
        if isinstance(x, list):
            x = torch.cat(x, dim=1)  # dim is (b, block_length*c, h, w)
        b, c, h, w = x.shape
        y_hold = y.clone()
        summed_y = torch.sum(y, dim=1, keepdim=True)
        y = y.argmax(dim=1, keepdim=True)  # b, c, h, w
        x = x.permute(0, 2, 3, 1).view(b * h * w, c)
        y = y.permute(0, 2, 3, 1).view(b * h * w, 1)
        # clouds aren't one-hot, just 0s
        y_mask = summed_y.permute(0, 2, 3, 1).view(b * h * w, 1)
        chosen_indices = torch.where(y_mask)[0]
        # print(x.shape, x.index_select(0, chosen_indices).shape); asdf
        if self.training:
            # extend memory
            if self.memory_x is not None:
                self.memory_x = torch.cat([self.memory_x, x.index_select(0, chosen_indices)])
                self.memory_y = torch.cat([self.memory_y, y.index_select(0, chosen_indices)])
            else:
                self.memory_x = x.index_select(0, chosen_indices).clone()
                self.memory_y = y.index_select(0, chosen_indices).clone()
            one_hot_preds = self.eye.to(x.device).index_select(0, torch.zeros(y.shape[0], dtype=torch.long,
                                                                              device=x.device))
        else:
            memory_indices = torch.randint(x.shape[0], (min(self.memory_x.shape[0], 10000),))
            memory_x_sample = self.memory_x[memory_indices]
            memory_y_sample = self.memory_y[memory_indices]

            if self.metric == "L2":
                dist_mat = pairwise_distances(x, memory_x_sample)
            elif self.metric == "dtw":
                dist_mat = torch.zeros(x.shape[0], memory_x_sample.shape[0])
                for i in range(x.shape[0]):
                    if not i % 1000: print(i)
                    for j in range(memory_x_sample.shape[0]):
                        dist_mat[i][j] = dtw(x[i].detach().cpu().numpy(), memory_x_sample[j].detach().cpu().numpy())
            else:
                raise NotImplementedError
            values, indices = torch.topk(dist_mat, self.k, dim=1, largest=False)
            label_sets = memory_y_sample[indices]
            preds = torch.mode(label_sets, dim=1).values
            # preds = y
            one_hot_preds = self.eye.to(x.device).index_select(0, preds.squeeze())
            print(one_hot_preds.sum(dim=0))
        one_hot_preds = one_hot_preds.view(b, h, w, -1).permute(0, 3, 1, 2)
        return one_hot_preds
        # return y_hold # one_hot_preds


def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist
