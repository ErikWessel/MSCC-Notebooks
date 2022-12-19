# Notebooks
This repository contains Jupyter-Notebooks that either make use of the existing system of microservies or generate data that is not yet available.

`sensor_data_generator.ipynb` generates missing data for the [Ground Data Service](https://git.scc.kit.edu/master-thesis-ai-ml-based-support-for-satellite-exploration/ground-data-service)

`locations_and_grid_cells_viewer.ipynb` leverages the running microservies [Ground Data Service](https://git.scc.kit.edu/master-thesis-ai-ml-based-support-for-satellite-exploration/ground-data-service) and [Satellite Data Service](https://git.scc.kit.edu/master-thesis-ai-ml-based-support-for-satellite-exploration/satellite-data-service) to create a visualization of a map of geo-spacial positions and the grid-cells that they are contained in.

## Note
Results of the execution of Jupyter-Notebooks should automatically be placed inside a `results/` directory to separate them from the notebooks and to prevent committing these files.