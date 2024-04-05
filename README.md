# Time-series inversion of a synthetic displacement signal

This repository is associated with:

Mueting, A.; Bookhagen, B. Assessing the accuracy of time-series inversion for reconstructing surface-displacement signals using Sentinel-2 and PlanetScope imagery (in prep.)

and constains scripts and examples for reconstructing a displacement signal from a network of pairwise displacement measurements.  

All relevant python functions can be found in [timeseries_inversion.py](./timeseries_inversion.py).
We show a processing example for a basic reconstruction and the effect of different noise types in this [Jupyter Notebook](./timeseries_inversion_basic_example.ipynb).
The effect of a disconnected network and the addition of indirect additions is described in a this [Jupyter Notebook](./timeseries_inversion_disconnected_networks.ipynb)
Finally, examples how to visualize the networks in form of a [graph](./Rplotting/plot_graph.R) or [correlation matrix](./Rplotting/plot_matrix.R) in R are contained in the [Rplotting](./Rplotting) folder.
