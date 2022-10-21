import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_seed(args):
    if args.seed[8:] == 'venus01':
        random.seed(int(args.seed[:7]))
        np.random.seed(int(args.seed[:7]))
        torch.manual_seed(int(args.seed[:7]))
    else:
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))

def clean():
    torch.cuda.empty_cache()
    print("finished clean!")

def saveModel(args,model, optimizer, F1, AP, Precision, Recall, path):
    state={'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'F1': F1,
    'AP': AP,
    'Precision':Precision,
    'Recall':Recall,
    'seed':args.seed,
    }
    torch.save(state,path)

def load_checkpoint(args,filename,model, optimizer):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename,map_location=args.device)
        # epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model,optimizer
    else:
        raise Exception("=> no checkpoint found at '{}'".format(filename))

class focal_loss(nn.Module):
    def __init__ ( self, alpha=0.25, gamma=2, num_classes=3, size_average=True ):
        """
        focal_loss function, -α(1-yi)**γ *ce_loss(xi,yi)
        This class implements focal_loss.
        :param alpha:   α,class weight.      when alpha is a list, it corresponds to the weight of each class. When it is a constant, the class weight is [alpha, 1-alpha, 1-alpha, ...].It is set as 0.25 in retainnet.
        :param gamma:   γ, used to adjust the focus on difficult and easy samples, it is set as 2 in retainnet.
        :param num_classes:     the number of class
        :param size_average:    the way of computing loss, by default is average
        """
        #super(focal_loss, self).__init__()
        super().__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α can be a list,size:[num_classes], will assign the weights to each of class
            print(" --- Focal_loss alpha = {}, will assign the weights to each of class --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # if alpha is constant, reduce the weight of the first class
            print(" --- Focal_loss alpha = {} ,will depriotirized background class, which should be used in detections--- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α is [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward ( self, preds, labels ):
        """
        focal_loss
        :param preds:   size:[B,N,C] or [B,C], corresponds to detections and classification, B: batch, N: the number of anchors, C: the number of classes
        :param labels:  size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # implemented nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss_out = loss.mean()
        else:
            loss_out = loss.sum()
        return loss_out