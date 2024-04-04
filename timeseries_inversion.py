#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import networkx as nx

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


def find_indirect(direct, full, dispcol = "disp", max_day_diff = 1e6):
    
    full  = full[["date0", "date1"]]
    direct = direct[["date0", "date1", dispcol]]
    
    merge = full.merge(direct, how='outer', indicator=True)
    indirect = merge[merge['_merge'] == 'left_only']
    indirect = indirect.drop(columns=['_merge'])
    
    if len(indirect) + len(direct) != len(full):
        print(f"Number of direct and indirect matches should sum up to {len(full)}.")
        return 
    
    out = []
    for i, row in indirect.iterrows():
        #step 1: find all connections in direct that fully cover that timespan
        covering = direct[(direct.date0 <= row.date0) & (direct.date1 >= row.date1)].copy()
        if len(covering)>0:
            covering["dt"] = covering.date1 - covering.date0
            rdt = (row.date1-row.date0).days
            #step 2: find shortest connection -> more likely that displacement can be linearly interpolated
            dmin = covering["dt"].min()

            if dmin.days-rdt <= max_day_diff:
                selected = covering[covering["dt"] == dmin].iloc[0] #use the first in case there are multiple options available
                #step3 when reference displacement is chosen, load dx and dy, calculate how much displacement there was 
                #per day and use that to estimate how much displacement has taken place over the timespan of the indirect connection in question
                d_soll = selected[dispcol]
                
                row["disp"] = d_soll / dmin.days * rdt

                row["date0_direct"] = selected.date0
                row["date1_direct"] = selected.date1              
                
                out.append(row)
          
    out = pd.DataFrame(out)
    out["connection_type"] = "indirect"
    direct["connection_type"] = "direct"       
    direct["date0_direct"] = np.nan
    direct["date1_direct"] = np.nan 

    out = pd.concat([out, direct])

    return out    


def find_indirect_minimum(direct, dispcol = "disp"):
    
    direct = direct[["date0", "date1", dispcol]]
    direct_old = direct.copy()

    G = nx.from_pandas_edgelist(direct, 'date0', 'date1')
    connected_components = list(nx.connected_components(G))
    #map group id to orig df
    group_mapping = {node: group_id for group_id, nodes in enumerate(connected_components) for node in nodes}
    direct['group_id'] = direct['date0'].apply(lambda x: group_mapping[x])
    
    nr_groups = len(direct.group_id.unique())
    group_counts = direct.group_id.value_counts().sort_values().reset_index()
    group_counts.columns = ["group_id", "count"]
    
    attempts = 0
    indirect = []
    while (len(group_counts)>1) and (attempts < nr_groups): #continue until there is only one group
        current = direct.loc[direct.group_id == group_counts.group_id.iloc[0]]
        other = direct.loc[direct.group_id != group_counts.group_id.iloc[0]]
        
        top_picks = []
        for _,conn in current.iterrows(): #search in other groups for suitable matches
            inside = other.loc[(other.date0 > conn.date0) &(other.date0 < conn.date1)].copy()
            if len(inside)>0:
                inside["dt"] =  inside.date0 - conn.date0
                selected = inside.loc[inside["dt"] == min(inside["dt"])].iloc[0]
                new_conn = conn.copy()
                new_conn.date1 = selected.date0
                new_conn[dispcol] = conn[dispcol] / (conn.date1-conn.date0).days * (new_conn.date1-new_conn.date0).days
                new_conn["date0_direct"] = conn.date0
                new_conn["date1_direct"] = conn.date1
                new_conn.group_id = selected.group_id
                new_conn["score"] = (selected["dt"].days + (conn.date1 -conn.date0).days)*conn[dispcol] #lowest number will be best connection = shortest direct and indirect
                top_picks.append(new_conn)
        top_picks = pd.DataFrame(top_picks)
        top_pick = top_picks.loc[top_picks["score"] == min(top_picks["score"])].iloc[0]
        
        indirect.append(top_pick)
        attempts += 1
        direct.loc[direct.group_id == group_counts.group_id.iloc[0], 'group_id'] = top_pick.group_id #merge current to selected group
        
        #recalculate group counts
        group_counts = direct.group_id.value_counts().sort_values().reset_index()
        group_counts.columns = ["group_id", "count"]

        
    indirect = pd.DataFrame(indirect)    
    indirect["connection_type"] = "indirect"
    direct_old["connection_type"] = "direct"     
    indirect["noise"] = np.nan
    indirect = indirect.drop(["score"], axis = 1)
    direct_old["date0_direct"] = np.nan
    direct_old["date1_direct"] = np.nan 
    out = pd.concat([indirect, direct_old])

    return(out)


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