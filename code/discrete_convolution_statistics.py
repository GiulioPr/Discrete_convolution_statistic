#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 00:14:37 2019

@author: Giulio Prevedello
"""

#LIBRARY DEPENDENCIES:
import numpy as np
import scipy.linalg as spl
import scipy.stats as sps
from functools import reduce

def mult_cov(p):
    return np.diag(p) - np.outer(p,p)

def conv_op(v, d):
    #V = np.asarray([np.zeros(d+len(v)-1) for times in range(d)])
    #for k in range(d):
    #    V[k][k:k+len(v)] = v
    V = np.array([np.hstack([np.zeros(k), v, np.zeros(d-1-k)]) for k in range(d)])
    return V.T 

def conv_covar(est_x, c_x):
    cov_x = list(map(mult_cov, est_x))
    TT_x = [
        conv_op(reduce(np.convolve, est_x[:k] + est_x[k+1:]),
                len(est_x[k])) for k in range(len(est_x))
    ]
    return sum(list(map(lambda TT,S,c:np.dot(TT, np.dot(S,TT.T))*c, TT_x, cov_x, c_x)))

def chi2_gof(o_freq, e_freq):
    indx = e_freq!=0.
    stat = (np.power(o_freq[indx] - e_freq[indx], 2.) / e_freq[indx]).sum()
    dof = indx.sum() - 1
    pval = sps.chi2.sf(stat, df=dof)
    return (stat, pval)

def chi2_ed(o_tab):
    pc = np.sum(o_tab, axis=0)
    pr = np.sum(o_tab, axis=1)
    e_tab = np.outer(pr, pc)/ o_tab.sum()
    indx = e_tab!=0.
    stat = (np.power(o_tab[indx] - e_tab[indx], 2.) / e_tab[indx]).sum()
    dof = indx.sum() - len(pc) - len(pr) + 1
    pval = sps.chi2.sf(stat, df=dof)
    return (stat, pval)

def low_rank_pinv(A, rk, rk_tol=np.finfo(float).resolution):
    (eig, P) = spl.eigh(A)
    tol_rk = max((eig > rk_tol).sum(), 1)
    if rk == -1:
        auto_rk = tol_rk
    else:
        auto_rk = min(rk, tol_rk)
    eig[:-auto_rk] = np.zeros(len(eig)-auto_rk)
    eig[-auto_rk:] = 1. / eig[-auto_rk:]
    return (np.dot(P, np.dot(np.diag(eig), P.T)), auto_rk)

def conv_test_true_S(sigma, ary_obsx, ary_obsy=[], gof_z=[], rk=-1, bool_force_rank=False):
    n_x = np.array([sum(obs) for obs in ary_obsx])
    est_x = [ary_obsx[k] / n_x[k] for k in range(len(ary_obsx))]
    if len(ary_obsy) != 0:
        n_y = np.array([sum(obs) for obs in ary_obsy])
        est_y = [ary_obsy[k] / n_y[k] for k in range(len(ary_obsy))]
    if len(gof_z) != 0:
        v = (reduce(np.convolve, est_x) - gof_z) * (min(n_x)**0.5)
    else:
        v = (reduce(np.convolve, est_x) - reduce(np.convolve, est_y)) * (min(min(n_x), min(n_y))**0.5)
    mp_sigma, auto_rk = low_rank_pinv(sigma, int(rk))######
    stat = reduce(np.dot, [v, mp_sigma, v.T])
    if bool_force_rank and rk > 0:
        pval = sps.chi2.sf(stat, df=rk)
        return (stat, pval)
    else:
        pval = sps.chi2.sf(stat, df=auto_rk)
        return (stat, pval, auto_rk)

def conv_test(ary_obsx, ary_obsy=[], gof_z=[], rk=-1, bool_force_rank=False):
    if len(ary_obsy) + len(gof_z) == 0:
        return (np.nan, np.nan)
    n_x = np.array([sum(obs) for obs in ary_obsx])
    if len(ary_obsy) != 0:
        n_y = np.array([sum(obs) for obs in ary_obsy])
        n_den = float(min(min(n_x), min(n_y)))
        c_y = n_den/n_y
    else:
        n_den = float(min(n_x))
    c_x = n_den/n_x
    est_x = [ary_obsx[k]/float(n_x[k]) for k in range(len(ary_obsx))]
    sigma = conv_covar(est_x[:], c_x)
    if len(ary_obsy) != 0:
        est_y = [ary_obsy[k]/float(n_y[k]) for k in range(len(ary_obsy))]
        if len(est_y) == 1:
            sigma = sigma + mult_cov(est_y[0])*c_y[0]
        else:
            sigma = sigma + conv_covar(est_y[:], c_y)
    if (sigma==np.zeros(np.shape(sigma))).all():
        if len(ary_obsy) == 0:
            o_freq = min(n_x) * reduce(np.convolve,est_x)
            e_freq = min(n_x) * gof_z
            return chi2_gof(o_freq, e_freq)
        else:
            o_tab = np.array([min(n_x) * reduce(np.convolve,est_x),
                              min(n_y) * reduce(np.convolve,est_y)])
            return chi2_ed(o_tab)
    else:
        mp_sigma, auto_rk = low_rank_pinv(sigma, int(rk))######
        if len(gof_z) != 0:
            v = (reduce(np.convolve,est_x) - gof_z) * (n_den**0.5)
        else:
            v = (reduce(np.convolve,est_x) - reduce(np.convolve,est_y)) * (n_den**0.5)
        stat = reduce(np.dot, [v, mp_sigma, v.T])
        if bool_force_rank and rk > 0:
            pval = sps.chi2.sf(stat, df=rk)
            return (stat, pval)
        else:
            pval = sps.chi2.sf(stat, df=auto_rk)
            return (stat, pval, auto_rk)


