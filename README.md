# Time-series inversion of a synthetic displacement signal

This repository constains scripts and examples for reconstructing a displacement signal from a network of pairwise displacement measurements through time-series inversion. Inversion is part of the processing chain for deriving a continuos time series from satellite-based measurements of displacements over landslides, glaciers, dunes or other Earth-surface processes. Here, we explore the impact of measurement errors and network connectivity using an artificial displacement signal.     

## Content
All relevant python functions can be found in [timeseries_inversion.py](./timeseries_inversion.py). We also created three Jupyter Notebooks that show their application with regards to the following topics: 
 - [Notebook 1](./timeseries_inversion_basic_example.ipynb): basic reconstruction and the effect of different types of measurement errors.
 - [Notebook 2](./timeseries_inversion_with_weights.ipynb): different weighting strategies that can help to improve the reconstruction accuracy.
 - [Notebook 3](./timeseries_inversion_sparse_and_disconnected_networks.ipynb): inversion of sparsely connected (one group but limited number of connections) and disconnected (seperate groups) networks.

## Installation
To install all necessary Python packages, create a new environment using conda and the provided [environment.yml](./environment.yml) file: 
```
conda env create -f environment.yml
conda activate ts_inversion
jupyter notebook
```

## R Plotting
In addition to the Python code, this repository contains examples for visualizing networks and inverted time series in R: 
- [network plot](./Rplotting/plot_graph.R)
- [correlation matrix](./Rplotting/plot_matrix.R) 
- [time-series with matrix inset](./Rplotting/matrix_inset.R)
All R scripts are contained in the [Rplotting](./Rplotting) folder.


## Citation

This repository is associated with:

Mueting, A., Charrier, L., and Bookhagen, B.: Assessing the accuracy of time-series inversion for reconstructing surface-displacement signals using Sentinel-2 and PlanetScope imagery (in prep.)
