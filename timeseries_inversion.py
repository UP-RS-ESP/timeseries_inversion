#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
from itertools import combinations
from matplotlib.patches import Arc
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.sparse as sp
import scipy.sparse.linalg
import networkx as nx
import pvlib
import rasterio
from tqdm import tqdm
import os
from multiprocessing import Pool
import math
from scipy.optimize import curve_fit
##########################################################################################################################
#preprocessing 

def pre_process_for_inversion(files, blocksize = 256, pix_to_m = True, medshift = True, unstable_mask = None, overwrite = False, ensure_same_size = True):
    
    files_tiled = []
    if ensure_same_size: 
        size_info = []
        #loop over all files to get their sizes to make sure they all have the same size
        for file in files:
            if os.path.isfile(file):
                with rasterio.open(file) as src: 
                    meta = src.meta
                    size_info.append({"filename": file,
                                      "width": meta["width"],
                                      "height": meta["height"]})
                    
        
        df = pd.DataFrame(size_info)
        df = df.sort_values(by=["height", "width"], ascending=[False, False]) #find max size for reference
        #TODO: select max, pad with nan
    
    for file in files: 
        if not os.path.isfile(file):
            print(f'File {file} not found')
            continue
        outname = f'{file[:-4]}_'
        if medshift: 
            outname += 'medshift_'
        if pix_to_m:
            outname += 'in_m_'
        outname += 'tiled.tif'
            
        if (not os.path.isfile(outname)) or overwrite: 
            with rasterio.open(file) as src:
                profile = src.profile # to be updated for storing the raster later
                meta = src.meta
                res = meta['transform'][0]
                bands = meta['count']
                
                if bands > 3: 
                    print(f'Not sure how to interpret this number of bands. Expected 1-3 bands (3rd band = good pixel mask), got: {bands}.')
                    continue
                
                data = {}
                
                for b in range(1, bands+1):
                    data[f'band_{b}'] = src.read(b)
                
            if bands == 3: # assuming input is a 3-band raster from ASP
                print('Interpreting and applying band 3 as good pixel mask.')
                
                data['band_1'][data['band_3'] == 0] = np.nan            
                data['band_2'][data['band_3'] == 0] = np.nan   
                del data['band_3']
            
            if medshift:
                if unstable_mask is not None:
                    with rasterio.open(unstable_mask) as src:
                        mask = src.read(1)
                        data['band_1'] -= np.nanmedian(data['band_1'][mask != 1])
                        data['band_2'] -= np.nanmedian(data['band_2'][mask != 1])
                        
                else: 

                        data['band_1'] -= np.nanmedian(data['band_1'])
                        data['band_2'] -= np.nanmedian(data['band_2'])
                        
            if pix_to_m: 
                
                for band in data:
                    data[band] *= res
                
            #save file
            profile.update(dtype='float32', tiled=True, blockxsize=blocksize, blockysize=blocksize, compress='deflate', count = len(data))
            
            with rasterio.open(outname, "w", **profile) as dst:
                for band in range(1, len(data)+1):
                    dst.write(data[f'band_{band}'], band)     
                print(f'I have written {outname}.')
                files_tiled.append(outname)
    return files_tiled
        

###########################################################################################################################
# basic inversion 

def mu_regularisation(regu:str, A:np.array, dates_range:np.array)->np.array:
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
    Construction of the design matix A in the formulation AX = B using the leap frog approach. Code provided by Laurane Charrier.
    
    :param dates: np array, where each line is (date1, date2) for which a velocity is computed (it corresponds to the original displacements)
    :param dates_range: list, dates of estimated displacemements in X
    
    :return: The design matrix A which represent the temporal closure of the displacement measurement network
    """
    # Search at which index in dates_range is stored each date in dates
    date1_indices = np.searchsorted(dates_range, dates[:, 0])
    date2_indices = np.searchsorted(dates_range, dates[:, 1]) - 1

    A = np.zeros((dates.shape[0], dates_range[1:].shape[0]), dtype='int32')
    for b in range(dates.shape[0]):
        A[b, date1_indices[b]:date2_indices[b] + 1] = 1

    return A

def create_design_matrix_cumulative_displacement(num_pairs, dates0, dates1):
    '''
    Create designmatrix connecting the cumulative displacement vector and the displacement measured between each date pair. 
    Alternative to Construction_A_LF() solving directly for cumulative displacement.
    Args: 
        num_pairs: integer value corresponding to the number of pairwise displacement measurements in network. 
        dates0: array of reference dates. 
        dates1: array of secondary dates. 
    Returns: 
        A: design matrix to be used in the inversion. 
    '''
   
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


def Inversion_A_LF(A, data, solver, Weight, mu, coef=1, ini=None, result_quality=None,
                   verbose=False):
    '''
    Invert the system AX = B for one component of the velocity, using a given solver. Code provided by Laurane Charrier.

    :param A: Matrix of the temporal inversion system AX = B
    :param data: np array, displacement observation B
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
    if isinstance(Weight, int): 
        if Weight ==1: 
            Weight = np.ones(v.shape[0]) #there is no weight, it corresponds to Ordinary Least Square        
    if solver == 'LSMR':
        F = np.vstack([np.multiply(Weight[Weight != 0][:, np.newaxis], A[Weight != 0]), F_regu]).astype('float')
        D = np.hstack([np.multiply(Weight[Weight != 0], v[Weight != 0]), D_regu]).astype('float')
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
 
    
def solve_matrix_system(A, B, weights = None):
    '''
    Solve matrix system of the form AX = B using least squares without a regularization term (only suited for fully connected networks)
    Args: 
        A: design matrix
        B: vector storing pairwise displacement measurements
        weights: vector with the weight to be assigned to every pairwise displacement measurement.
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

    X, _, _, _ = np.linalg.lstsq(A.astype(np.float64), B, rcond=1e-10)
    ts[1:, 0] = X.astype(np.float32)[:,0] #first displacement will be 0

    return ts[:,0]


def run_inversion(net, dispcol = 'disp', weightcol = None, regu = False):
    '''
    Wrapper for running the time-series inversion for the given network.
    Args: 
        net: pandas dataframe storing the network to be inverted. Must contain the columns date0, date1 and displacement.
        dispcol: string indicating the name of the column containing displacement measurements. Default = 'disp'
        weightcol: string indicating the name of the column in net storing weights for the individual connections. Default = None, i.e. no weights.
        regu: boolean indicating if the inversion shall be solved using a regularization term. Default = False, i.e. no regularization. 
    Returns: 
        out: pandas dataframe containing the result of the inversion. Columns include dates and cumulative displacement (disp_inversion)
    
    '''
    
    num_pairs = len(net)
    net = net.sort_values(['date0', 'date1']).reset_index(drop = True)

    print('Number of image pairs: %d'%num_pairs)
        
    if not regu: #classic inversion 
        net = net.copy()
        if type(net.date0.iloc[0]) == pd._libs.tslibs.timestamps.Timestamp:
            net.date0 = net.date0.dt.date
            net.date1 = net.date1.dt.date
            
        dates0 = np.asarray([datetime.strptime(str(x), '%Y-%m-%d') for x in net.date0])
        dates1 = np.asarray([datetime.strptime(str(x), '%Y-%m-%d') for x in net.date1])
            
        displacements = np.array(net[dispcol]).reshape((num_pairs, 1))
    
        # create design_matrix
        A, date_list = create_design_matrix_cumulative_displacement(num_pairs, dates0, dates1)
    
        nIslands = np.min(A.shape) - np.linalg.matrix_rank(A)
        print(f'Number of groups in network: {nIslands +1}')
        
        if weightcol != None:
            weights = net[weightcol]
        else:
            weights = None
         
        #run inversion
        timeseries = solve_matrix_system(A, displacements, weights = weights)
      
        out = pd.DataFrame({'date': date_list, 'disp_inversion': timeseries})
        out['date'] = pd.to_datetime(out.date)
        
        return out

    else: # add a regularization term and solve the inversion
        
        data = net[['date0','date1']].values
        sample_dates = pd.concat([net.date0, net.date1]).unique()
        sample_dates = np.sort(sample_dates)
    
        dates_range = Construction_dates_range_np(data)
        A = Construction_A_LF(data,dates_range)
        nIslands = np.min(A.shape) - np.linalg.matrix_rank(A)
        print(f'Number of groups in network: {nIslands +1}')
        print("Solving the inversion including a regularization term ...")
        mu = mu_regularisation(regu=1, A=A, dates_range=sample_dates)
    
        if weightcol is not None:
            Weight = net[weightcol].values
        else: 
            Weight = 1
        timeseries,normresidual = Inversion_A_LF(A, net[dispcol].values, solver='LSMR', Weight=Weight, mu=mu, coef=1, ini=None, result_quality=None,
                           verbose=False)
        timeseries_cumulative = np.cumsum(timeseries) #build the cumulative time series bc LF design matrix solves for displacement at each time step
        timeseries_cumulative = np.insert(timeseries_cumulative, 0, 0, axis=0) #set first date to zero
        
        out = pd.DataFrame({'date': sample_dates, 'disp_inversion': timeseries_cumulative})
        out['date'] = pd.to_datetime(out.date)
        
        return out

#############################################################################################################################
# parallelize inversion to speed up raster processing
    
def prep_inversion_parallel(net, verbose = False):
    
    num_pairs = len(net)
    net = net.sort_values(['date0', 'date1']).reset_index(drop = True)

    data = net[['date0','date1']].values
    sample_dates = pd.concat([net.date0, net.date1]).unique()
    sample_dates = np.sort(sample_dates)

    dates_range = Construction_dates_range_np(data)
    A = Construction_A_LF(data,dates_range)
    
    if verbose: 
        nIslands = np.min(A.shape) - np.linalg.matrix_rank(A)
        print('Number of image pairs: %d'%num_pairs)
        print(f'Number of groups in network: {nIslands +1}')
        print("Solving the inversion including a regularization term ...")
    mu = mu_regularisation(regu=1, A=A, dates_range=sample_dates)
    
    return A, mu, sample_dates
    
def run_inversion_parallel(disp, A, mu, sample_dates, weights = None):

    if weights is not None:
        Weight = weights
    else: 
        Weight = 1
    timeseries,normresidual = Inversion_A_LF(A, disp, solver='LSMR', Weight=Weight, mu=mu, coef=1, ini=None, result_quality=None,
                       verbose=False)
    timeseries_cumulative = np.cumsum(timeseries) #build the cumulative time series bc LF design matrix solves for displacement at each time step
    timeseries_cumulative = np.insert(timeseries_cumulative, 0, 0, axis=0) #set first date to zero
    
    out = timeseries_cumulative
    
    return out

def process_window_inversion(args):
    
    window, A, mu, sample_dates, network, band, weighted = args

    z_stack = []
    for f in network.filename:
        with rasterio.open(f) as src:
            data = src.read(band, window=window)
            z_stack.append(data)
    z_stack = np.stack(z_stack, axis=0) 

    h, w = z_stack.shape[1:]
    result = np.full((h, w, len(sample_dates)), np.nan, dtype=np.float32)
    weights = None 
    
    for i in range(h):
        for j in range(w):
            disp_series = z_stack[:, i, j]

            if np.isnan(disp_series).all():
                result[i, j, :] = np.nan
            
            
            elif np.isnan(disp_series).any():
                valid_mask = ~np.isnan(disp_series)
                if valid_mask.sum() < 2:
                    result[i, j, :] = np.nan
                    continue

                d_sub = disp_series[valid_mask]
                net_sub = network[valid_mask]
                A_sub, mu_sub, sample_dates_sub = prep_inversion_parallel(net_sub)
                
                if weighted: 
                    weights = net_sub["weight"].to_numpy()
                out = run_inversion_parallel(d_sub, A_sub, mu_sub, sample_dates_sub, weights)
                
                if len(sample_dates_sub) != len(sample_dates): #if enough values are missing, maybe not all dates are present and ts is shorter

                    full_series = np.full(len(sample_dates), np.nan, dtype=np.float32)
                    date_idx = pd.Index(sample_dates).get_indexer(sample_dates_sub)

    
                    full_series[date_idx] = out
                    result[i, j, :] = full_series
                else: 
                    result[i, j, :] = out

            else: #all data present
                if weighted: 
                    weights = network["weight"].to_numpy()
                    
                out = run_inversion_parallel(disp_series, A, mu, sample_dates, weights)
                result[i, j, :] = out

    return result


def inversion_tiled(network, band, cpu, ext = "", weights = None):
        
    if weights is not None: 
        network["weight"] = weights
        weighted = True
    else: weighted = False
    
    network = network.sort_values(["date0", "date1"]).reset_index(drop=True)
    files = network["filename"].tolist()
    
    A, mu, sample_dates = prep_inversion_parallel(network) #precalculate values for when everything is present
    
    with rasterio.open(files[0]) as src:
        if not src.is_tiled:
            print("Input is not tiled!")
            return
        block_height, block_width = src.block_shapes[0] #take first band as reference
        width, height = src.width, src.height
        windows = [
            rasterio.windows.Window(x, y, block_width, block_height)
            for y in range(0, height, block_height)
            for x in range(0, width, block_width)
        ]
    args = [(window, A, mu, sample_dates, network, band, weighted) for window in windows]

    #run inversion for every tile
    with Pool(processes=cpu) as pool: 
            results = list(tqdm(pool.imap(process_window_inversion, args), total=len(windows)))
            
    #reassemble results
    full_result = np.full((len(sample_dates), height, width), np.nan, dtype=np.float32)
    
    for window, result_block in zip(windows, results):
        row_off, col_off = int(window.row_off), int(window.col_off)
        h, w = result_block.shape[:2]
        full_result[:, row_off:row_off+h, col_off:col_off+w] = result_block.transpose(2, 0, 1)
    
    with rasterio.open(files[0]) as src:
        profile = src.profile
    
    profile.update({
        "count": len(sample_dates),
        "dtype": "float32",
        "compress": "deflate",                    
        "tiled": True
    })
    
    #save
    if not os.path.isdir('./inversion_out'):
        os.path.mkdir('./inversion_out')
        
    if band == 1: 
        output_path = f"./inversion_out/inversion{ext}_dx.tif"
    if band == 2: 
        output_path = f"./inversion_out/inversion{ext}_dy.tif"
    
    with rasterio.open(output_path, "w", **profile) as dst:
        for i in range(len(sample_dates)):
            dst.write(full_result[i, :, :], i + 1)
            

def extract_stats_in_mask(file, mask):
    
    with rasterio.open(file) as src: #read all bands into memory
        inv = src.read()

    with rasterio.open(mask) as src:
        m = src.read(1)
            
    m_selected = m == 1  

    inv_reshaped = inv.reshape(inv.shape[0], -1)  

    selected_pixels = inv_reshaped[:, m_selected.flatten()] 

    stats = pd.DataFrame({"median":np.nanmedian(selected_pixels, axis=1),
                          "p25":np.nanpercentile(selected_pixels, 0.25, axis=1),
                          "p75":np.nanpercentile(selected_pixels, 0.75, axis=1), 
                          "mean":np.nanmean(selected_pixels, axis=1),
                          "std":np.nanstd(selected_pixels, axis=1)})
    return stats



##########################################################################################################################
# seasonal bias estimation & mitigation 

def sine_wave_fixed_freq(t, A, phi, C):
    wavelength = 365.25
    f = 1 / wavelength  #fixed frequency = 1 year
    return A * np.sin(2 * np.pi * f * t + phi) + C # A = amplitude, t = time, phi = phase shift, c = vertical offset 

def fit_sine_per_pixel(ts, sample_dates, daily_index):

    if np.isnan(ts).all(): 
        return np.array([np.nan, np.nan, np.nan])

    #linearly interpolate displacement values for trend estimation (moving average)
    s = pd.Series(ts, index=sample_dates)
    si = s.reindex(daily_index).interpolate("linear").ffill().bfill()
    trend_daily = si.rolling(window=396, center=True, min_periods=1).mean()

    #get trend for original timestepd
    trend_orig = trend_daily.reindex(sample_dates).interpolate("linear")
    #detrend
    detr = ts - trend_orig.to_numpy()
    
    #mask remaining nans
    mask = ~np.isnan(detr)
    if mask.sum() < 4:  # need at least 4 data points for stable sine fit
        return np.array([np.nan, np.nan, np.nan])

    detr_valid = detr[mask]
    dates_valid = sample_dates[mask]

    #fit sine
    dsdt = (dates_valid - dates_valid[0]).days.to_numpy()
    try:
        popt, _ = curve_fit(
            sine_wave_fixed_freq,
            dsdt,
            detr_valid,
            p0=[1, 0, 0],
            bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), #ensure amplitude is positive
            maxfev=2000)

        A, phi, C = popt
    except (RuntimeError):
        A, phi, C = np.nan, np.nan, np.nan

    return np.array([A, phi, C])

    
def process_window_sine_fit(args):
    
    window, sample_dates, file = args
    daily_index = pd.date_range(sample_dates.min(), sample_dates.max(), freq="D")
    with rasterio.open(file) as src:
        chunk = src.read(window=window)
        h, w  = chunk.shape[1:]

    result = np.full((3, h, w), np.nan)
    
    for i in range(h):
        for j in range(w):
            ts = chunk[:, i, j]
            result[:, i, j] = fit_sine_per_pixel(ts, sample_dates, daily_index)
    
    return result


def fit_sine_tiled(file, sample_dates, cpu):

    with rasterio.open(file) as src:
        if not src.is_tiled:
            print("Input is not tiled!")
            #return
        block_height, block_width = src.block_shapes[0] #take first band as reference
        width, height = src.width, src.height
        windows = [
            rasterio.windows.Window(x, y, block_width, block_height)
            for y in range(0, height, block_height)
            for x in range(0, width, block_width)
        ]
        meta = src.meta.copy()
    

    fit_full = np.full((3, height, width), np.nan, dtype="float32") #store amp, phase and c
    
    if isinstance(sample_dates, np.ndarray): #need sample dates to be a datetime index
        sample_dates = pd.to_datetime(sample_dates)
        
    args = [(window, sample_dates, file) for window in windows]

    with Pool(processes=cpu) as pool: 
            results = list(tqdm(pool.imap(process_window_sine_fit, args), total=len(windows)))


    for window, result_block in zip(windows, results):
        fit_full[:, window.row_off : window.row_off + window.height,
                 window.col_off : window.col_off + window.width] = result_block
        
    #convert phase to degrees
    fit_full[1,:,:] = np.degrees(np.mod(fit_full[1,:,:], 2 * np.pi))

    #save
    meta.update(
        count       = 3,    
        dtype       = "float32",
        nodata      = np.nan,
        driver      = "GTiff",
        compress    = "deflate")
    
    with rasterio.open(file[:-4] + "_sine_fit.tif", "w", **meta) as dst:
        dst.write(fit_full)  

def get_sun_pos(lon, lat, datetime):

    
    sun_az = pvlib.solarposition.get_solarposition(
        time=datetime,
        latitude=lat,
        longitude=lon
    )['azimuth'].values[0]
    
    sun_el = pvlib.solarposition.get_solarposition(
        time=datetime,
        latitude=lat,
        longitude=lon
    )['elevation'].values[0]
    
    
    return(sun_az, sun_el)

##########################################################################################################################

# experiments with synthetic networks and displacement signals
    
def find_dsoll(row, signal, datecol = 'date', dispcol = 'disp'):
    '''
    Find the displacement that should have taken place between two dates according to the given reference signal.
    Args: 
        row: row of a pandas dataframe for which the displacement shall be estimated. 
        signal: pandas dataframe containing the reference displacement signal based on which pairwise measurements will be estimated. 
        datecol: string indicating the name of the column in the signal df containing the dates. Default = date
        dispcol: string indicating the name of the column in the signal df containing the reference displacement. Default = disp. 
    Returns: 
        d_soll: displacement value according to provided reference signal. 
    '''
    
    dp0 = signal[signal[datecol] == row.date0]
    dp1 = signal[signal[datecol] == row.date1]
    
    if (len(dp0) > 0) and (len(dp1) > 0):
        d_soll = (dp1[dispcol].iloc[0] - dp0[dispcol].iloc[0])
    else: 
        d_soll =  np.nan
        
    return d_soll

    
def min_max_scaler(x):
    '''
    Scales the values in the given input array or pd.Series between 0 and 1.
    '''
    return (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))


def create_disconnected_groups(nr_groups, dates, overlap = 0):
    '''
    Turns given dates into a network with disconnected groups. 
    Args: 
        nr_groups: integer number specifying number of groups to be created.
        dates: pd Datetime index or array of dates in network. 
        overlap: integer number indicating the temporal overlap between groups in units of timesteps. 
        By default, there will be no temporal overlap, i.e. overlap = 0. 
    Returns: 
        df: pandas dataframe with date pairs and corresponding group_id.
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
    Args: 
        nr_groups: integer number specifying number of groups to be created.
        dates: pd Datetime index or array of dates in network. 
        seed: seed for random group assignment (optional).
    Returns: 
        df: pandas dataframe with date pairs and corresponding group_id.

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

##########################################################################################################################
# visualization

def plot_timeseries(timeseries, original_signal = None, legend = []):
    '''
    Plot inverted time series and residual to original signal.
    Args: 
        timeseries: pandas dataframe retrieved from the inversion process storing dates and displacement OR
                    list of pandas dataframes, e.g. [df1, df2, df3] to plot the results from multiple inversions together.
        original_signal: Optional. pandas dataframe storing the dates and displacement of the original signal or any signal that 
                    the inverted timeseries shall be compared against (residuals are calculated).
        legend: Optional. List of strings with names of legend items. Must be of same length as the provided number of time series. 
    '''

    colormap = cm.get_cmap('Dark2')
    if type(timeseries) == list:
        
        #check for provided legend
        if len(legend) == 0 or len(legend) != len(timeseries):
            legend = [f"Time series #{ts+1}" for ts in range(len(timeseries))]
            
        colors = [colormap(i) for i in np.arange(0,len(timeseries))]
        
    else: 
        timeseries = [timeseries]
        colors = [colormap(0)]
        if len(legend) == 0 or len(legend) != 1:
            legend = ["Times series #1"]


    if original_signal is not None:
        original_signal = original_signal.copy()
        original_signal.date = pd.to_datetime(original_signal.date)
        original_signal.columns = ["date", "disp_true"]
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (16, 6))
        
    else: 
        fig, ax1 = plt.subplots(1,1, figsize = (8, 6))
        ax1.axhline(y=0, color='gray', linestyle = '--')
        
    if original_signal is not None: 
        timeseries = [pd.merge(ts, original_signal, on = 'date', how = 'left') for ts in timeseries]
        ax1.axhline(y=0, color='gray', linestyle = '--')
        ax1.plot(original_signal.date, original_signal.disp_true, label = "True displacement", color = "gray")
        
    
    for i in range(len(timeseries)): 
        ax1.plot(timeseries[i].date, timeseries[i].disp_inversion, color = colors[i], label = legend[i])
        ax1.scatter(timeseries[i].date, timeseries[i].disp_inversion, color = colors[i])
        
    ax1.legend()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Displacement')
    ax1.set_title('Cumulative displacement time-series')

    if original_signal is not None: 
        ax2.axhline(y=0, color='gray', linestyle = '--')
        for i in range(len(timeseries)): 
            ax2.plot(timeseries[i].date, timeseries[i].disp_true-timeseries[i].disp_inversion, color = colors[i], label = legend[i])
        ax2.set_title('Residual')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Residual')
        ax2.legend()
        
    plt.tight_layout()
    plt.show()


def plot_network(network, outname = None):
    '''
    Plot network structure as arc diagramm. 
    Args: 
        network: df with pairwise displacement measurements between date0 and date1
    '''
    network = network.copy()
    #need to convert dates to numeric values for plotting
    if type(network.date0.iloc[0]) == pd._libs.tslibs.timestamps.Timestamp:

        network['date0'] = pd.to_datetime(network['date0']).dt.date
        network['date1'] = pd.to_datetime(network['date1']).dt.date

    all_dates = sorted(set(network['date0']).union(set(network['date1'])))

    #mapping for dates
    numeric_dates = {date: (date - min(all_dates)).days for date in all_dates}

    network['num_date0'] = network['date0'].map(numeric_dates)
    network['num_date1'] = network['date1'].map(numeric_dates)
    
    network["num_diff"] = network.num_date1 - network.num_date0
    
    #check connectivity of network
    G = nx.from_pandas_edgelist(network, 'date0', 'date1')
    ccs = list(nx.connected_components(G))
    group_mapping = {node: group_id for group_id, nodes in enumerate(ccs) for node in nodes}
    network['group_id'] = network['date0'].apply(lambda x: group_mapping[x])
    
    #get colors for unique groups
    colormap = cm.get_cmap('tab10')
    colors = {group_id: colormap(i) for i, group_id in enumerate(network.group_id.unique())}

    fig, ax = plt.subplots(figsize=(10, 4))

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
        
    legend = [Line2D([0], [0], color=colors[group_id], lw=1, label=f'Group {group_id+1}') for group_id in network.group_id.unique()]
    ax.legend(handles=legend, loc = 1)

    ax.set_yticks([]) #empty y axis
    ax.set_xticks(list(numeric_dates.values())) 
    ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in all_dates], rotation=90, size = 7) #overwrite x axis labels
    
    ax.set_ylim(-20, network.num_diff.max()/2 +20)
    
    #remove boundaries
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    if outname is not None:
        plt.savefig(outname, dpi = 300)
    plt.show()
    
    
def plot_timesteps(file, dates = None):
    
    '''Plots every timestep of the inversion result (raster). Provide filename and optionally list of sample dates.'''
    with rasterio.open(file) as src: #read all bands into memory
        inv = src.read()
        
    bands = inv.shape[0]
    rows = 4
    cols = math.ceil(bands/rows)
    
    vmax = np.nanpercentile(inv[bands-1,:,:], 99) #use 99th percentile of last timestamp as color lim
    
    fig, ax = plt.subplots(rows, cols, figsize=(3*cols, 3*cols), squeeze=False, constrained_layout=True)
    
    ax = ax.ravel()
    
    for i in range(bands):
        im = ax[i].imshow(inv[i, :, :], vmin = -vmax, vmax = vmax, cmap = "coolwarm")
        if dates is not None:
            ax[i].set_title(dates[i])
    
    # remove empty panels
    for j in range(bands, len(ax)):
        ax[j].axis("off")
    
    cbar = fig.colorbar(im, ax=ax, location="bottom", shrink=0.8, fraction=0.025, pad=0.02)

    cbar.set_label("Cumulative Displacement")
    
    plt.show()
        
def plot_sine_fit(file):
    '''Plots the results of the per-pixel sine fit estimating the magnitude of seasonal biases.'''
    
    with rasterio.open(file) as src: 
        amp = src.read(1)
        phs = src.read(2)
        c = src.read(3)
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    im0 = ax[0].imshow(amp, vmin=0, vmax=np.percentile(amp, 99), cmap="Reds")
    im1 = ax[1].imshow(phs, vmin=0, vmax=360, cmap="hsv")
    im2 = ax[2].imshow(c,   vmin=0, vmax=np.percentile(c, 99), cmap="Blues")
    
    titles = ["Amplitude", "Phase", "Vertical offset"]
    ims = [im0, im1, im2]
    
    for a, im, title in zip(ax, ims, titles):
        a.set_title(title)
        divider = make_axes_locatable(a)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        
    plt.tight_layout()
    plt.show()
