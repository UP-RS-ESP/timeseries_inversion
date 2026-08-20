# Surface Displacement Reconstruction via Time-Series Inversion and Seasonal Bias Mitigation

This repository contains scripts and examples for reconstructing a displacement signal from a network of pairwise displacement measurements through time-series inversion. Temporal inversion is part of the processing chain for deriving a continuous time series from satellite-based measurements of displacements over landslides, glaciers, dunes or other Earth-surface processes. Prerequisite are several temporally overlapping displacement fields obtained from image cross-correlation. The inversion process returns a multi-band raster with the cumulative displacement estimated at each time step. In addition, this repository provides functionalities to correct for seasonal biases which are common in cross-season image pairs (different illumination) of mountainous terrain and presents a challenge for the identification of seasonally driven displacement.

## Content
Please refer to [demo.ipynb](demo.ipynb) for a full walk-through of the time-series inversion process. All core functionality is implemented in [timeseries_inversion.py](./timeseries_inversion.py). 


## Installation
To install all necessary Python packages, create a new environment using conda and the provided [environment.yml](./environment.yml) file: 
```
conda env create -f environment.yml
conda activate ts_inversion
```


## Citation

This repository is associated with:

Mueting, A., Charrier, L., and Bookhagen, B.: Challenges in reconstructing seasonally driven landslide motion from optical satellite data: insights from the Del Medio catchment, NW Argentina, EGUsphere (preprint), [https://doi.org/10.5194/egusphere-2025-6445](https://doi.org/10.5194/egusphere-2025-64459), 2026. 
