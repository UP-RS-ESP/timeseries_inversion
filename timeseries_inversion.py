#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
from datetime import datetime
import networkx as nx
from functools import reduce

# define neccessary functions
def find_dsoll(row, signal, datecol = 'date', dispcol = 'disp'):
    
    '''Find the displacement that took place between two dates according to the given reference signal.'''
    
    dp0 = signal[signal[datecol] == row.date0]
    dp1 = signal[signal[datecol] == row.date1]
    
    if (len(dp0) > 0) and (len(dp1) > 0):
        d_soll = (dp1[dispcol].iloc[0] - dp0[dispcol].iloc[0])
    else: 
        d_soll =  np.nan
        
    return d_soll


def create_design_matrix_cumulative_displacement(num_pairs, dates0, dates1):
    
    '''Create designmatrix connecting the cumulative displacement vector and the displacement measured between each date pair'''
   
    unique_dates = np.union1d(np.unique(dates0), np.unique(dates1))
    num_dates = len(unique_dates)

    datepair_list = []
    for i in range(len(dates0)):
        datepair_list.append('%s_%s'%(datetime.strftime(dates0[i], '%Y%m%d'), datetime.strftime(dates1[i], '%Y%m%d')))

    A = np.zeros((num_pairs, num_dates), np.float32)

    date_list = list(unique_dates)
    date_list = [datetime.strftime(d, '%Y%m%d') for d in date_list]

    for i in range(num_pairs):
        ind1, ind2 = (date_list.index(d) for d in datepair_list[i].split('_'))
        A[i, ind1] = -1
        A[i, ind2] = 1

    # Remove reference date as it can not be resolved
    ref_date = datetime.strftime(min(dates0),'%Y%m%d')

    ind_r = date_list.index(ref_date)
    A = np.hstack((A[:, 0:ind_r], A[:, (ind_r+1):]))

    return A, date_list


def solve_matrix_system(A, B, rcond=1e-10, weights = None):
    
    '''
    Solve matrix system using least squares. 
    Args: 
        A: design matrix
        B: vector storing pairwise displacement measurements
    Returns: 
        ts: inverted time series
    '''
    
    if weights is not None: 
        weights = np.diag(weights).astype(np.float64)
        A = np.dot(weights, A.astype(np.float64))
        B = np.dot(weights, B)

    num_dates = A.shape[1] + 1

    ts = np.zeros((num_dates, 1), dtype=np.float32) #intialize empty output time series

    B = B.astype(np.float64)

    X, _, _, _ = np.linalg.lstsq(A.astype(np.float64), B, rcond=rcond)
    ts[1:, 0] = X.astype(np.float32)[:,0] #first displacement will be 0

    return ts[:,0]


def run_inversion(net, weightcol = None):
    
    '''Run time-series inversion for the given network dataframe'''
    
    net = net.copy()
    if type(net.date0[0]) == pd._libs.tslibs.timestamps.Timestamp:
        net.date0 = net.date0.dt.date
        net.date1 = net.date1.dt.date
        
    dates0 = np.asarray([datetime.strptime(str(x), '%Y-%m-%d') for x in net.date0])
    dates1 = np.asarray([datetime.strptime(str(x), '%Y-%m-%d') for x in net.date1])
        
    num_pairs = len(net)
    displacements = np.array(net.disp).reshape((num_pairs, 1))

    # create design_matrix
    A, date_list = create_design_matrix_cumulative_displacement(num_pairs, dates0, dates1)
    
    print('Number of image pairs: %d'%num_pairs)
    nIslands = np.min(A.shape) - np.linalg.matrix_rank(A)
    print('Number of disconnected components in network: %d '%nIslands)
    if nIslands > 1:
        print('\tThe network appears to be disconnected. Consider connecting it to avoid artifacts.')
    
    if weightcol != None:
        weights = net[weightcol]
    else:
        weights = None
     
    #run inversion
    timeseries = solve_matrix_system(A, displacements, rcond=1e-10, weights = weights)
  
    out = pd.DataFrame({'date': date_list, 'disp_inversion': timeseries})
    out['date'] = pd.to_datetime(out.date)
    return out



def plot_results(timeseries, original_signal = None):
    
    '''
    Plot inverted time series and residual to original signal.
    Plots data from multiple inversion runs when parameter timeseries is provided as a list of pd dfs.
    '''
    
    original_signal = original_signal.copy()
    
    colormap = cm.get_cmap('Dark2')
    if type(timeseries) == list:
        # rename columns
        timeseries = [t.copy() for t in timeseries] #make a hard copy to avoid col renaming in orig dfs
        for i in range(len(timeseries)):
            timeseries[i].columns = ["date", f"disp_inversion_{i}"]
        timeseries = reduce(lambda left, right: pd.merge(left, right, on="date"), timeseries)
     
        colors = [colormap(i) for i in np.arange(0,len(timeseries))]
    else: 
        colors = [colormap(0)]
                                                   

    if original_signal is not None:
        original_signal.date = pd.to_datetime(original_signal.date)
        original_signal.columns = ["date", "disp_true"]
        fig, ax = plt.subplots(1,2, figsize = (16, 6))
    else: 
        fig, ax = plt.subplots(1,1, figsize = (8, 6))
        
    timeseries = pd.merge(timeseries, original_signal, on = 'date', how = 'left')
    
    ax[0].plot(original_signal.date, original_signal.disp_true, label = "True displacement", color = "gray")
    for i, col in enumerate(timeseries.columns.drop(["date", "disp_true"])): 
        ax[0].plot(timeseries.date, timeseries[col], label = col, color = colors[i])
        ax[0].scatter(timeseries.date, timeseries[col], color = colors[i])
        
    ax[0].legend()
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Displacement')
    ax[0].set_title('Cumulative displacement time-series')

    if original_signal is not None: 
        ax[1].axhline(y=0, color='gray')
        for i, col in enumerate(timeseries.columns.drop(["date", "disp_true"])): 
            ax[1].plot(timeseries.date, timeseries.disp_true-timeseries[col], color = colors[i])
        ax[1].set_title('Residual')
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('Residual')
    
def min_max_scaler(x):
    return (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))