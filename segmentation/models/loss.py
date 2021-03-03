# Insipiration: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, with_logits=True, reduce=True, weight=None):
        """
        Implements Focal Loss for binary classification (i.e. assumes targets are one-hot).
        FocalLoss is detailed: https://arxiv.org/pdf/1708.02002.pdf

        :param alpha: Class imbalance term
        :param gamma: Used to change focus on misclassified examples
        :param with_logits: Whether inputs are in logit format or probability format
        :param reduce: Whether to get mean loss
        :param weight: Weight value per class with which to multiply loss; size (c,)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.with_logits = with_logits
        self.reduce = reduce
        self.register_buffer('weight', weight)

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') \
            if self.with_logits else F.binary_cross_entropy(inputs, targets, reduction='none')

        pt = (-bce_loss).exp()
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss

        # Need self.weight to be of shape self.inputs.shape[-1]
        if self.weight is not None:
            focal_loss = focal_loss.permute(0, 2, 3, 1)  # shift #c to last dim
            focal_loss *= self.weight

        return torch.mean(focal_loss) if self.reduce else focal_loss


class BatchCriterion(nn.Module):
    ''' Compute the loss within each batch
    '''
    def __init__(self, negM=1, T=0.1, num_sample_points=1024):
        super(BatchCriterion, self).__init__()
        self.negM = negM
        self.T = T
        self.num_sample_points = num_sample_points
        self.diag_mat = 1 - torch.eye(2 * num_sample_points).cuda()
        
    def forward(self, x1, x2):
        """
        x and targets are both feature maps
        n x c x h x w
        want to make them
        nhw x c
        And then select indices to operate on


        In original version: https://github.com/mangye16/Unsupervised_Embedding_Learning/blob/master/BatchAverage.py
        the features are concatenated along dim=0 when fed in
        so first half of them corresponds with order in second half
        I can just do the same then let original code handle things
        batchSize will end up being double points_to_sample
            TODO Reduce to one variable
        """
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        # Now these are in order n x h x w x c
        # can just flatten up until last dim
        # TODO: Wrap this in function
        x1 = x1.flatten(start_dim=0, end_dim=2)
        x2 = x2.flatten(start_dim=0, end_dim=2)
        selected_indices = torch.randperm(x1.size(0))[:self.num_sample_points]
        x1 = x1[selected_indices]
        x2 = x2[selected_indices]
        x = torch.cat( (x1,x2), dim=0)
        batchSize = x.size(0)
        norm = x.pow(2).sum(1, keepdim=True).pow(1./2)
        x = x.div(norm)
        #get positive innerproduct
        reordered_x = torch.cat((x.narrow(0,batchSize//2,batchSize//2),\
                x.narrow(0,0,batchSize//2)), 0)
        #reordered_x = reordered_x.data
        pos = (x*reordered_x.data).sum(1).div_(self.T).exp_()

        #get all innerproduct, remove diag
        all_prob = torch.mm(x,x.t().data).div_(self.T).exp_()*self.diag_mat
        if self.negM==1:
            all_div = all_prob.sum(1)
        else:
            #remove pos for neg
            all_div = (all_prob.sum(1) - pos)*self.negM + pos

        lnPmt = torch.div(pos, all_div)
        # negative probability
        Pon_div = all_div.repeat(batchSize,1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)

        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
        lnPonsum = lnPonsum * self.negM
        loss = - (lnPmtsum + lnPonsum)/batchSize
        return loss
