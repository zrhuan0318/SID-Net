import numpy as np
import itertools
import torch
from collections import defaultdict

def mylog(x):
    """Safe log2 computation, avoiding 0, NaN and Inf values"""
    valid_indices = (x > 0) & (~np.isnan(x)) & (~np.isinf(x))
    log_values = np.zeros_like(x, dtype=float)
    log_values[valid_indices] = np.log2(x[valid_indices])
    return log_values

def myhistogram(Y, nbins=5):
    """
    Build the joint histogram (joint probability distribution).
    Automatically switches between numpy and torch:
      - If number of variables <= 32, use numpy.histogramdd
      - If number of variables > 32, use torch.bincount
    
    Parameters:
        Y: array (nvars, nsamples), rows = variables, columns = samples
        nbins: number of bins for discretization
    Returns:
        hist: joint probability distribution (ndarray)
    """
    nvars, nsamples = Y.shape

    # -------- numpy implementation (<=32 dimensions) --------
    if nvars <= 32:
        hist, _ = np.histogramdd(Y.T, bins=nbins)
        hist += 1e-14
        hist /= hist.sum()
        return hist

    # -------- torch implementation (>32 dimensions) --------
    X = torch.tensor(Y.T, dtype=torch.float32)  # shape (nsamples, nvars)

    # normalize to [0,1]
    X_min = X.min(dim=0, keepdim=True)[0]
    X_max = X.max(dim=0, keepdim=True)[0]
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)

    # digitize
    X_bins = (X_norm * nbins).long().clamp(max=nbins-1)

    # compute unique index for each sample
    powers = nbins ** torch.arange(nvars-1, -1, -1)
    indices = (X_bins * powers).sum(dim=1)

    # count frequencies
    hist_flat = torch.bincount(indices, minlength=nbins**nvars).float()

    # reshape into n-dimensional histogram
    hist = hist_flat.reshape([nbins] * nvars)

    # normalize
    hist += 1e-14
    hist /= hist.sum()

    return hist.numpy()

def compute_all_mi_combinations(Y, nbins=5, max_combs=2):
    """
    SURD-based information decomposition (Unique, Redundant, Synergistic).
    Based on specific mutual information allocation.

    Input:
        Y: array (nvars, nsamples), first row = target, remaining rows = input variables
        nbins: number of bins for discretization
        max_combs: maximum combination order (e.g. 2 = only pairwise)
    Output:
        I_U: array, unique information of each variable
        I_R: dict, redundant contributions
        I_S: dict, synergistic contributions
        MI: dict, mutual information of each variable combination
    """
    Ntot = Y.shape[0]   # target + agents
    tot_inds = range(1, Ntot)

    # marginal distribution of target
    p_target = myhistogram(Y[0, :].reshape(1, -1), nbins)
    nbins_target = p_target.shape[0]

    combs, Is = [], {}
    # iterate over all combinations (1, 2, ..., max_combs)
    for r in range(1, max_combs + 1):
        for combo in itertools.combinations(tot_inds, r):
            combs.append(combo)
            # build histogram for [target, combo]
            selected_vars = (0,) + combo
            hist = myhistogram(Y[selected_vars, :], nbins)
            # p(target, combo)
            p_joint = hist
            # p(combo)
            p_combo = hist.sum(axis=0)
            # conditional probability p(target|combo)
            p_target_expand = p_target.reshape(-1, *(1,) * len(combo))
            p_target_given_combo = p_joint / (p_combo + 1e-14)
            # specific mutual information
            s_info = (p_target_given_combo * 
                      (mylog(p_target_given_combo) - mylog(p_target_expand))
                     ).sum(axis=tuple(range(1, len(selected_vars))))
            Is[combo] = s_info

    # mutual information (for each combination)
    MI = {k: (Is[k] * p_target.squeeze()).sum() for k in Is.keys()}

    # initialize redundancy / synergy / unique
    I_R = {cc: 0 for cc in combs}
    I_S = {cc: 0 for cc in combs if len(cc) >= 2}
    I_U = np.zeros(Ntot - 1)

    # iterate over each target bin and distribute specific MI
    for t in range(nbins_target):
        I1 = np.array([ii[t] for ii in Is.values()])
        i1 = np.argsort(I1)
        lab = [combs[i_] for i_ in i1]
        lens = np.array([len(l) for l in lab])

        # update specific MI (zero out smaller values if dominated by larger sets)
        I1 = I1[i1]
        for l in range(1, lens.max()):
            inds_l2 = np.where(lens == l+1)[0]
            Il1max = I1[lens == l].max()
            inds_ = inds_l2[I1[inds_l2] < Il1max]
            I1[inds_] = 0

        i1 = np.argsort(I1)
        lab = [lab[i_] for i_ in i1]

        Di = np.diff(I1[i1], prepend=0.)
        red_vars = list(tot_inds)

        # allocate to R / U / S
        for i_, ll in enumerate(lab):
            info = Di[i_] * p_target.squeeze()[t]
            if len(ll) == 1:
                I_R[ll] += info
                I_U[ll[0]-1] += info
                if ll[0] in red_vars:
                    red_vars.remove(ll[0])
            else:
                I_S[ll] += info

    return I_U, I_R, I_S, MI
