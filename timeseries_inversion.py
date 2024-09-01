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
from itertools import combinations
from matplotlib.patches import Arc
from matplotlib.lines import Line2D
import scipy.sparse as sp
import scipy.sparse.linalg



def mu_regularisation(regu, A, dates_range):
    """
    Compute the Tikhonov regularisation matrix. Code provided by Laurane Charrier.

    :param regu: str, type of regularization
    :param A: np array, design matrix
    :param dates_range: list, list of estimated dates
    :param ini: initial parameter (velocity and/or acceleration mean)

    :return mu: Tikhonov regularisation matrix
    """

    # First order Tikhonov regularisation
    if regu == 1:
        mu = np.identity(A.shape[1], dtype='float32')
        mu[np.arange(mu.shape[0] - 1) + 1, np.arange(mu.shape[0] - 1)] = -1
        mu /= (np.diff(dates_range) / np.timedelta64(1, 'D'))
        mu = np.delete(mu, 0, axis=0)

    # First order Tikhonov regularisation, with an apriori on the acceleration
    elif regu == '1accelnotnull':
        mu = np.diag(np.full(A.shape[1], -1, dtype='float32'))
        mu[np.arange(A.shape[1] - 1), np.arange(A.shape[1] - 1) + 1] = 1
        mu /= (np.diff(dates_range) / np.timedelta64(1, 'D'))
        mu = np.delete(mu, -1, axis=0)

    return mu


def Construction_dates_range_np(data):
    """
    Construction of the dates of the estimated displacement in X with an irregular temporal sampling (ILF).
    Code provided by Laurane Charrier.
    :param data: np.ndarray, an array where each line is (date1, date2, other elements) for which a velocity have been mesured
    :return: np.ndarray, the dates of the estimated displacement in X
    """

    dates = np.concatenate([data[:, 0], data[:, 1]])  # concatante date1 and date2
    dates = np.unique(dates)  # remove duplicates
    dates = np.sort(dates)  # Sort the dates
    return dates


def Construction_A_LF(dates, dates_range):
    """
    Construction of the design matix A in the formulation AX = Y. Code provided by Laurane Charrier.
    
    :param dates: np array, where each line is (date1, date2) for which a velocity is computed (it corresponds to the original displacements)
    :param dates_range: list, dates of estimated displacemements in X
    
    :return: The design matrix A which represent the temporal closure of the displacement measurement network
    """
    # Search at which index in dates_range is stored each date in dates
    date1_indices = np.searchsorted(dates_range, dates[:, 0])
    date2_indices = np.searchsorted(dates_range, dates[:, 1]) - 1

    A = np.zeros((dates.shape[0], dates_range[1:].shape[0]), dtype='int32')
    for y in range(dates.shape[0]):
        A[y, date1_indices[y]:date2_indices[y] + 1] = 1

    return A


def Inversion_A_LF(A, data, solver, Weight, mu, coef=1, ini=None, result_quality=None,
                   verbose=False):
    '''
    Invert the system AX = Y for one component of the velocity, using a given solver. Code provided by Laurane Charrier.

    :param A: Matrix of the temporal inversion system AX = Y
    :param data: np array, displacement observation Y
    :param solver: 'LSMR', 'LSMR_ini', 'LS', 'LS_bounded', 'LSQR'
    :param Weight: Weight, =1 for Ordinary Least Square
    :param mu: regularization matrix
    :param coef: Coef of Tikhonov regularization
    :param ini: np array, Initialization of the inversion
    :param: result_quality: None or list of str, which can contain 'Norm_residual' to determine the L2 norm of the residuals from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate each value in X (it corresponds to A.dot(weight))
    :param regu : str, type of regularization

    :return X: The ILF temporal inversion of AX = Y using the given solver
    :return residu_norm: Norm of the residual (when showing the L curve)
    '''

    v = data
    D_regu = np.zeros(mu.shape[0])
    F_regu = np.multiply(coef, mu)
    if Weight == 1: Weight = np.ones(v.shape[0]) #there is no weight, it corresponds to Ordinary Least Square
    if solver == 'LSMR':
        F = np.vstack([np.multiply(Weight[Weight != 0][:, np.newaxis], A[Weight != 0]), F_regu]).astype('float32')
        D = np.hstack([np.multiply(Weight[Weight != 0], v[Weight != 0]), D_regu]).astype('float32')
        F = sp.csc_matrix(F)  # column-scaling so that each column have the same euclidian norme (i.e. 1)
        X = sp.linalg.lsmr(F, D)[0]  # If atol or btol is None, a default value of 1.0e-6 will be used. Ideally, they should be estimates of the relative error in the entries of A and b respectively.

    elif solver == 'LSMR_ini':  # 50ms
        # 16.7 ms ± 141 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        condi = (Weight != 0)
        W = Weight[condi]
        F = sp.csc_matrix(
            np.vstack([np.multiply(W[:, np.newaxis], A[condi]),
                       F_regu]))  # stack ax and regu, and remove rows with only 0
        D = np.hstack([np.multiply(W, v[condi]), D_regu])  # stack ax and regu, and remove rows with only

        if ini.shape[0] == 2:  # if only the average of the entire time series
            x0 = np.full(len(A.shape [1]) - 1, ini, dtype='float32')
        else:
            x0 = ini
        X = sp.linalg.lsmr(F, D, x0=x0)[0]

    if result_quality is not None and 'Norm_residual' in result_quality:  # to show the L_curve
        R_lcurve = F.dot(X) - D  # 50.7 µs ± 327 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
        residu_norm = [np.linalg.norm(R_lcurve[:np.multiply(Weight[Weight != 0], v[Weight != 0]).shape[0]], ord=2),
                       np.linalg.norm(R_lcurve[np.multiply(Weight[Weight != 0], v[Weight != 0]).shape[0]:] / coef,
                                      ord=2)]
    else:
        residu_norm = None

    return X, residu_norm
 
    

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


def run_inversion(net, weightcol = None, regu = False):
    '''Wrapper for running the time-series inversion for the given network dataframe'''
    
    num_pairs = len(net)
    print('Number of image pairs: %d'%num_pairs)
        
    if not regu: #classic inversion 
        net = net.copy()
        if type(net.date0[0]) == pd._libs.tslibs.timestamps.Timestamp:
            net.date0 = net.date0.dt.date
            net.date1 = net.date1.dt.date
            
        dates0 = np.asarray([datetime.strptime(str(x), '%Y-%m-%d') for x in net.date0])
        dates1 = np.asarray([datetime.strptime(str(x), '%Y-%m-%d') for x in net.date1])
            
        displacements = np.array(net.disp).reshape((num_pairs, 1))
    
        # create design_matrix
        A, date_list = create_design_matrix_cumulative_displacement(num_pairs, dates0, dates1)
    
        nIslands = np.min(A.shape) - np.linalg.matrix_rank(A)
        print(f'Number of disconnected groups in network: {nIslands +1}')
        
        if weightcol != None:
            weights = net[weightcol]
        else:
            weights = None
         
        #run inversion
        timeseries = solve_matrix_system(A, displacements, rcond=1e-10, weights = weights)
      
        out = pd.DataFrame({'date': date_list, 'disp_inversion': timeseries})
        out['date'] = pd.to_datetime(out.date)
        
        return out

    else: # add a regularization term and solve the inversion
        
        data = net[['date0','date1']].values
        sample_dates = pd.concat([net.date0, net.date1]).unique().astype("datetime64[ns]").to_numpy()
        sample_dates.sort()
    
        dates_range = Construction_dates_range_np(data)
        A = Construction_A_LF(data,dates_range)
        nIslands = np.min(A.shape) - np.linalg.matrix_rank(A)
        print(f'Number of disconnected groups in network: {nIslands +1}')
        print("Solving the inversion including a regularization term ...")
        mu = mu_regularisation(regu=1, A=A, dates_range=sample_dates)
    
    
        timeseries,normresidual = Inversion_A_LF(A, net['disp'].values, solver='LSMR', Weight=1, mu=mu, coef=1, ini=None, result_quality=None,
                           verbose=False)
        timeseries_cumulative = np.cumsum(timeseries) #build the cumulative time series bc LF design matrix solves for displacement at each time step
        timeseries_cumulative = np.insert(timeseries_cumulative, 0, 0, axis=0) #set first date to zero
        
        out = pd.DataFrame({'date': sample_dates, 'disp_inversion': timeseries_cumulative})
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
        
    plt.tight_layout()
    plt.show()

    
def min_max_scaler(x):
    return (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))

def create_disconnected_groups(nr_groups, dates, overlap = 0):
    '''
    Turns given dates into a network with disconnected groups. Specify number of groups and temporal overlap as input parameters. 
    '''
    
    dates = np.array(dates)
    cut = int(len(dates)/nr_groups)
    
    if int(cut/2) < overlap:
        print(f"Overlap cannot be greater that group size. Please choose a value <= {int(cut/2) < overlap}!")
        return
    
    gids = []
    for g in range(nr_groups):
        gids = gids + [g]*(cut-overlap*2)  + [g+1]*overlap + [g]*overlap
        
    if len(gids) < len(dates): #add not fitting dates to last group
        gids = gids + [g]*(len(dates)-len(gids))
    
    #erase any higher numbers than nr_groups
    
    gids = np.array(gids)
    gids[gids >= nr_groups] = g
    
    #now make image pairs
    date_combinations = []
    group_ids = []
    for g in range(nr_groups):
        gcombinations = list(combinations(dates[gids == g], 2))
        date_combinations = date_combinations + gcombinations
        group_ids += [g] * len(gcombinations)
        
    df = pd.DataFrame(date_combinations, columns=['date0', 'date1'])
    df["group_id"] = group_ids

    return df


def create_random_groups(nr_groups, dates, seed = 123):
    '''
    Randomly assigns acquisitions to a given number of groups and only establishes connections between them. 

    '''
    np.random.seed(seed)
    dates = np.array(dates)
    gids = np.random.randint(nr_groups, size = len(dates))
    
    #now make image pairs
    date_combinations = []
    group_ids = []

    for g in range(nr_groups):
        gcombinations = list(combinations(dates[gids == g], 2))
        date_combinations = date_combinations + gcombinations
        group_ids += [g] * len(gcombinations)
        
    df = pd.DataFrame(date_combinations, columns=['date0', 'date1'])
    df["group_id"] = group_ids
    
    return df


def plot_network(network):
    network = network.copy()
    #need to convert dates to numeric values for plotting
    if type(network.date0[0]) == pd._libs.tslibs.timestamps.Timestamp:

        network['date0'] = pd.to_datetime(network['date0']).dt.date
        network['date1'] = pd.to_datetime(network['date1']).dt.date

    all_dates = sorted(set(network['date0']).union(set(network['date1'])))

    #mapping for dates
    numeric_dates = {date: (date - min(all_dates)).days for date in all_dates}

    network['num_date0'] = network['date0'].map(numeric_dates)
    network['num_date1'] = network['date1'].map(numeric_dates)
    
    network["num_diff"] = network.num_date1 - network.num_date0
    
    #get colors for unique groups
    colormap = cm.get_cmap('tab10')
    #mapping for colors
    colors = {group_id: colormap(i) for i, group_id in enumerate(network.group_id.unique())}

    fig, ax = plt.subplots(figsize=(8, 4))

    # plot nodes
    ax.scatter(list(numeric_dates.values()), [0] * len(numeric_dates), color='black')

    # plot arcs 
    for idx, row in network.iterrows():
        ax.add_patch(Arc(
            ((row['num_date0'] + row['num_date1']) / 2, 0), 
            row.num_diff,            
            row.num_diff,       
            #set theta only have half circle                             
            theta2=180,                                     
            color= colors[row.group_id]
        ))
        
    # Create legend lines
    legend = [Line2D([0], [0], color=colors[group_id], lw=1, label=f'Group {group_id}') for group_id in network.group_id.unique()]
    ax.legend(handles=legend, loc = 2)

    ax.set_yticks([]) #empty y axis
    ax.set_xticks(list(numeric_dates.values())) 
    ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in all_dates], rotation=90) #overwrite x axis labels
    
    ax.set_ylim(-20, network.num_diff.max()/2 +20)
    
    #remove boundaries
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    
    
