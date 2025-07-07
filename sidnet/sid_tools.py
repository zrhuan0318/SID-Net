# sid_tools.py
import numpy as np
import torch

def myhistogram(x, nbins):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(1)

    x_min = x.min(dim=0, keepdim=True)[0]
    x_max = x.max(dim=0, keepdim=True)[0]
    x_norm = (x - x_min) / (x_max - x_min + 1e-8)
    x_bins = (x_norm * nbins).long().clamp(max=nbins-1)

    powers = nbins ** torch.arange(x.shape[1]-1, -1, -1)
    indices = (x_bins * powers).sum(dim=1)
    hist_flat = torch.bincount(indices, minlength=nbins**x.shape[1]).float()
    hist = hist_flat.reshape([nbins] * x.shape[1])
    hist += 1e-14
    hist /= hist.sum()
    return hist.numpy()

def mylog(x):
    valid_indices = (x != 0) & (~np.isnan(x)) & (~np.isinf(x))
    log_values = np.zeros_like(x)
    log_values[valid_indices] = np.log2(x[valid_indices])
    return log_values

def entropy(p):
    return -np.sum(p * mylog(p))

def entropy_nvars(p, indices):
    excluded_indices = tuple(set(range(p.ndim)) - set(indices))
    marginalized_distribution = p.sum(axis=excluded_indices)
    return entropy(marginalized_distribution)

def cond_entropy(p, target_indices, conditioning_indices):
    joint_entropy = entropy_nvars(p, set(target_indices) | set(conditioning_indices))
    conditioning_entropy = entropy_nvars(p, conditioning_indices)
    return joint_entropy - conditioning_entropy

def mutual_info(p, set1_indices, set2_indices):
    entropy_set1 = entropy_nvars(p, set1_indices)
    conditional_entropy = cond_entropy(p, set1_indices, set2_indices)
    return entropy_set1 - conditional_entropy

def cond_mutual_info(p, ind1, ind2, ind3):
    combined_indices = tuple(set(ind2) | set(ind3))
    return cond_entropy(p, ind1, ind3) - cond_entropy(p, ind1, combined_indices)

def transfer_entropy(p, target_var):
    num_vars = len(p.shape) - 1
    TE = np.zeros(num_vars)
    for i in range(1, num_vars + 1):
        present_indices = tuple(range(1, num_vars + 1))
        conditioning_indices = tuple([target_var] + [j for j in range(1, num_vars + 1) if j != i and j != target_var])
        cond_ent_target_given_past = cond_entropy(p, (0,), conditioning_indices)
        cond_ent_target_given_past_and_input = cond_entropy(p, (0,), present_indices)
        TE[i-1] = cond_ent_target_given_past - cond_ent_target_given_past_and_input
    return TE
