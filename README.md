# Time-series inversion of a synthetic displacement signal

This repository contains scripts and examples for reconstructing a displacement signal from a network of pairwise displacement measurements through time-series inversion. Inversion is part of the processing chain for deriving a continuous time series from satellite-based measurements of displacements over landslides, glaciers, dunes or other Earth-surface processes. Prerequisite for the inversion are several temporally overlapping displacement fields obtained from image cross-correlation. The inversion process returns a multi-band raster with the cumulative displacement estimated at each unique acquisition time. In addition, this repository provides functionalities to correct for seasonal bias which is common in cross-season image pairs of mountainous terrain. 

## Content
Please refer to [demo.ipynb](demo.ipynb) for a full walk-through of the time-series inversion process. All core functionality is implemented in [timeseries_inversion.py](./timeseries_inversion.py). 


In addition, there are several notebooks that explore the effect of errors, network structure and sampling interval on the inversion results based on synthetic data. These can be found in the [artificial_examples](./artificial_examples/) folder. 


## Installation
To install all necessary Python packages, create a new environment using conda and the provided [environment.yml](./environment.yml) file: 
```
conda env create -f environment.yml
conda activate ts_inversion
jupyter notebook
```


## Citation

This repository is associated with:

Mueting, A., Charrier, L., and Bookhagen, B.: Reconstructing landslide motion from optical satellite imagery in seasonal climates: insights from the Del Medio catchment, NW Argentina (in prep.)
