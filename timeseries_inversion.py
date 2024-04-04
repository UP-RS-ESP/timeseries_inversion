#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

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


def solve_matrix_system(A, y, rcond=1e-10, weights = None):
    
    '''Solve matrix system using least squares'''
    
    if weights is not None: 
        weights = np.diag(weights).astype(np.float64)
        A = np.dot(weights, A.astype(np.float64))
        y = np.dot(weights, y)

    num_dates = A.shape[1] + 1

    ts = np.zeros((num_dates, 1), dtype=np.float32) #intialize empty output timeseries

    y = y.astype(np.float64)

    X, _, _, _ = np.linalg.lstsq(A.astype(np.float64), y, rcond=rcond)
    ts[1:, 0] = X.astype(np.float32)[:,0] #first displacement will be 0

    return ts[:,0]


def run_inversion(net, weightcol = None):
    
    '''Run time-series inversion for the given network dataframe'''
        
    dates0 = np.asarray([datetime.strptime(str(x), '%Y-%m-%d') for x in net.date0])
    dates1 = np.asarray([datetime.strptime(str(x), '%Y-%m-%d') for x in net.date1])
        
    num_pairs = len(net)
    displacements = np.array(net.disp).reshape((num_pairs, 1))

    # create design_matrix
    A, date_list = create_design_matrix_cumulative_displacement(num_pairs, dates0, dates1)
    
    print('Number of image pairs: %d'%num_pairs)
    nIslands = np.min(A.shape) - np.linalg.matrix_rank(A)
    print('Number of connected components in network: %d '%nIslands)
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


def plot_results(timeseries, signal):
    
    '''Plot inverted time series and residual to original signal'''
    
    signal.date = pd.to_datetime(signal.date)
    timeseries = pd.merge(timeseries, signal, on = 'date', how = 'left')
    timeseries.columns = ['date', 'disp_inversion', 'disp_true']

    fig, ax = plt.subplots(1,2, figsize = (16, 6))

    ax[0].plot(signal.date, signal.disp, c = 'gray', label = 'Original')
    ax[0].plot(timeseries.date, timeseries.disp_inversion, c = '#046C9A', label = 'Reconstruction')
    ax[0].scatter(timeseries.date, timeseries.disp_inversion, c = '#046C9A')
    ax[0].legend()
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Displacement')
    ax[0].set_title('Cumulative displacement time-series')

    ax[1].axhline(y=0, color='gray')
    ax[1].plot(timeseries.date, timeseries.disp_true-timeseries.disp_inversion)
    ax[1].set_title('Residual')
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Residual')