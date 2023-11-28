import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def accuracy(output, target):
    """
    Compute the accuracy of the model based on the outputs and targets.

    Parameters:
    output (torch.Tensor): The output of the model.
    target (torch.Tensor): The target.

    Returns:
    res (torch.Tensor): The accuracy.
    """
    # Get the batch size
    batch_size = target.size(0)
    # Get the index of the max log-probability
    _, pred = output.topk(1, 1, True, True)
    # Get the prediction
    pred = pred.t()
    # Get the number of correct predictions
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # Get the number of correct predictions in the batch
    correct_1 = correct[:1].view(-1).float().sum(0)
    # Compute the accuracy
    res = correct_1.mul_(100.0 / batch_size)
    return res

class SymmetricCrossEntropyLoss(nn.Module):
    """
    A custom loss function that combines the standard cross entropy loss and the reverse cross entropy loss.
    """
    def __init__(self, alpha=0.5, beta=1.5, num_classes=3):
        """
        Initialize the loss function.

        Parameters:
        alpha (float): The weight for the standard cross entropy loss. Default: 0.5.
        beta (float): The weight for the reverse cross entropy loss. Default: 1.5.
        num_classes (int): The number of classes. Default: 3.
        """
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
    
    def forward(self, logits, targets):
        """
        Define the forward pass of the loss function.
        
        Parameters:
        logits (torch.Tensor): The output of the model.
        targets (torch.Tensor): The target.
        """
        onehot_targets = torch.eye(self.num_classes).to(targets.device)[targets]
        # Standard Cross Entropy Loss
        ce_loss = F.cross_entropy(logits, targets)
        # Reverse Cross Entropy Loss
        rce_loss = (-onehot_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        return (self.alpha * ce_loss + self.beta * rce_loss) / (self.alpha + self.beta)

class CyclicalLR:
    """
    A custom learning rate scheduler that implements the cyclical learning rate policy.
    """
    def __init__(self, optimizer, r1=0.01, r2=0.001, c=10):
        """
        Initialize the learning rate scheduler.
        
        Parameters:
        optimizer (torch.optim.Optimizer): The optimizer.
        r1 (float): The minimum learning rate. Default: 0.01.
        r2 (float): The maximum learning rate. Default: 0.001.
        c (int): The cycle length. Default: 10.
        """
        self.optimizer = optimizer
        self.r1 = r1
        self.r2 = r2
        self.c = c
        # Initialize the current iteration
        self.t = 0

    def step(self):
        self.t += 1
        # Compute the learning rate
        s_t = (1 + ((self.t - 1) % self.c)) / self.c
        r_t = (1 - s_t) * self.r1 + s_t * self.r2

        # Set the learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = r_t

class ReweightLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, out, T, target):
        loss = 0.
        out_softmax = F.softmax(out, dim=1)
        for i in range(len(target)):
            temp_softmax = out_softmax[i]
            temp = out[i]
            temp = torch.unsqueeze(temp, 0)
            temp_softmax = torch.unsqueeze(temp_softmax, 0)
            temp_target = target[i]
            temp_target = torch.unsqueeze(temp_target, 0)
            pro1 = temp_softmax[:, target[i]] 
            out_T = torch.matmul(T.t(), temp_softmax.t())
            out_T = out_T.t()
            pro2 = out_T[:, target[i]] 
            beta = pro1 / pro2
            beta = Variable(beta, requires_grad=True)
            cross_loss = F.cross_entropy(temp, temp_target)
            _loss = beta * cross_loss
            loss += _loss
        return loss / len(target)

class ReweightingRevisionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, out, T, correction, target):
        loss = 0.
        out_softmax = F.softmax(out, dim=1)
        for i in range(len(target)):
            temp_softmax = out_softmax[i]
            temp = out[i]
            temp = torch.unsqueeze(temp, 0)
            temp_softmax = torch.unsqueeze(temp_softmax, 0)
            temp_target = target[i]
            temp_target = torch.unsqueeze(temp_target, 0)
            pro1 = temp_softmax[:, target[i]]
            T = T + correction
            T_result = T
            out_T = torch.matmul(T_result.t(), temp_softmax.t())
            out_T = out_T.t()
            pro2 = out_T[:, target[i]]    
            beta = (pro1 / pro2)
            cross_loss = F.cross_entropy(temp, temp_target)
            _loss = beta * cross_loss
            loss += _loss
        return loss / len(target)

def norm(T):
    row_sum = np.sum(T, 1)
    T_norm = T / row_sum
    return T_norm

def error(T, T_true):
    error = np.sum(np.abs(T-T_true)) / np.sum(np.abs(T_true))
    return error

def transition_matrix_generate(noise_rate=0.5, num_classes=3):
    P = np.ones((num_classes, num_classes))
    n = noise_rate
    P = (n / (num_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, num_classes-1):
            P[i, i] = 1. - n
        P[num_classes-1, num_classes-1] = 1. - n
    return P


def fit(X, num_classes, filter_outlier=False):
    # number of classes
    c = num_classes
    T = np.empty((c, c))
    eta_corr = X
    for i in np.arange(c):
        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], 97, interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)
        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]
    return T