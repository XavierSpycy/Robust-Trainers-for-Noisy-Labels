import torch
import numpy as np
import torch.nn.functional as F

class DualT(object):
    """
    Dual T: Reducing estimation error for transition matrix in label-noise learning
    """
    def estimate(self, model, estim_loader, num_classes=3,):
        """
        Define the estimation process of Dual T.

        Parameters:
        model (torch.nn.Module): The model to be evaluated.
        estim_loader (torch.utils.data.DataLoader): The data loader for estimation.
        num_classes (int, optional): The number of classes. Default: 3.

        Returns:
        dual_t_matrix (np.array): The estimated dual transition matrix. shape: (num_classes, num_classes).
        """
        # Set model to evaluation mode
        model.eval()
        # Estimate the transition matrix
        p = []
        T_spadesuit = np.zeros((num_classes, num_classes))
        with torch.no_grad():
            for i, (_, images, n_target) in enumerate(estim_loader):
                images = images.cuda()
                n_target = n_target.cuda()
                pred = model(images)
                probs = F.softmax(pred, dim=1).cpu().data.numpy()
                _, pred = pred.topk(1, 1, True, True)           
                pred = pred.view(-1).cpu().data
                n_target = n_target.view(-1).cpu().data
                for i in range(len(n_target)): 
                    T_spadesuit[int(pred[i])][int(n_target[i])]+=1
                p += probs[:].tolist()  
        T_spadesuit = np.array(T_spadesuit)
        sum_matrix = np.tile(T_spadesuit.sum(axis = 1),(num_classes,1)).transpose()
        T_spadesuit = T_spadesuit/sum_matrix
        p = np.array(p)
        T_clubsuit = self.est_t_matrix(p, filter_outlier=True)
        T_spadesuit = np.nan_to_num(T_spadesuit)
        dual_t_matrix = np.matmul(T_clubsuit, T_spadesuit)
        return dual_t_matrix
    
    def est_t_matrix(self, eta_corr, filter_outlier=False):
        """
        A helper function to estimate the transition matrix.

        Parameters:
        eta_corr (np.array): The estimated class probabilities for each example. shape: (num_examples, num_classes). 
        filter_outlier (bool): Whether to filter out outlier examples.

        Returns:
        T (np.array): The estimated transition matrix. shape: (num_classes, num_classes).
        """
        num_classes = eta_corr.shape[1]
        T = np.empty((num_classes, num_classes))

        # Find a 'perfect example' for each class
        for i in np.arange(num_classes):
            # find the example with the highest probability for class i
            if not filter_outlier:
                idx_best = np.argmax(eta_corr[:, i])
            else:
                eta_thresh = np.percentile(eta_corr[:, i], 97, interpolation='higher')
                robust_eta = eta_corr[:, i]
                robust_eta[robust_eta >= eta_thresh] = 0.0
                idx_best = np.argmax(robust_eta)
            # Set the transition matrix
            for j in np.arange(num_classes):
                T[i, j] = eta_corr[idx_best, j]
        return T