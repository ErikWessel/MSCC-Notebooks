# Notebooks
This repository contains the Jupyter-Notebooks of the corresponding thesis "AI/ML-based support of satellite sensing for cloud cover classification".
The notebooks are seperated according to specialized tasks.
The following section specifies the order of usage, when starting from scratch.

## Data Collection
The `data_collection.ipynb` notebook is responsible for collecting data from the microservices through the AIMLSSE-API.
Before running this notebook, make sure to create a `login.yml` file that has the following content:
```
username: <username>
password: <password>
```
The values should be replaced with login credentials for the [Copernicus Open Access Hub](https://scihub.copernicus.eu/dhus/#/home), since the service for satellite observations depends on access to this data source.

Next, the variable `target_state` is set to the **name of a U.S. state** for which the data should be collected.

The `station_radius_of_influence` needs to be set to the radius that should be contained in one outgoing image.
Per default, this is set to 16 kilometers.

Finally, the `server_ip` needs to be set to refer to the server that is hosting the AIMLSSE services for both **weather station data** and **satellite observations**.

Now you can run all cells of the notebook, as the data will be collected without any user interaction.
Intermediate visualizations are provided to give the user feedback about the status of the service.
Do to the nature of being a notebook, user action can at any point interrupt the download process to continue some other time.

Collected data is stored in the `data/queries/<target-state>/` subdirectory.
## Data Preprocessing

The `data_preprocessing.ipynb` notebook is responsible for analyzing, transforming, and visualizing data that is already collected.
Some prerequisites have to be considered before running the notebook.

The variable `target_states` is a **list of names of U.S. states** for which the data should be preprocessed.
Performing preprocessing on multiple states at the same time allows to create unified statistics and reduce the need for user-interaction.
The data is still preprocessed individually, by annotating the state that the entries belong to.

`max_abs_timedelta_minutes` is the maximum allowed absolute time difference between satellite observation and METARs.
The default is set to 30 minutes.

`remove_ambiguous_cloud_cover` removes METARs that contain ambiguous cloud cover, which stands for situations where multiple layers of clouds are given and the last layer is not complete coverage of the sky (OVC).

Now the notebook can be executed.

Preprocessing results are stored in the `labels.csv` files inside the preprocessing subdirectory of each target states' data (`data/queries/<target-state>/preprocessing/labels.csv`).
## Dataset Creation

The `dataset_creation.ipynb` notebook is responsible for creating a dataset from preprocessed data.
Some prerequisites have to be considered before running the notebook.

The variable `target_states` is a **list of names of U.S. states** for which the data should be combined into a dataset.

The `station_radius_of_influence` needs to be set to the radius that is used for the collected data.
Per default, this is set to 16 kilometers.
This value is necessary to compute the bounding geometry, creating clusters of stations to prevent leakage of data between the training, validation, and test sets.

The `image_size` value is equivalent to the image size in pixels (for width and height) that should be created for the dataset.
The default is set to 300x300 pixels.

`max_abs_timedelta_minutes` is the maximum allowed absolute time difference between satellite observation and METARs.
The default is set to 30 minutes.

`remove_ambiguous_cloud_cover` should be set to the same value as in the preprocessing and is only used for naming the dataset.

`make_distribution_uniform` creates a dataset with little to no imbalance by reducing the amount of data of all classes to the amount of the class with the least data.

Now the notebook can be executed.

The dataset is stored in the `data/` subdirectory by default.
## Machine Learning Training

The `ml_program.ipynb` notebook is responsible for training the machine learning model on the previously created dataset.
It requires `ml_commons.py` and `ml_config.yml`.
The notebook can also be directly converted to a python script for scheduling and automatic execution in the background.
Some prerequisites have to be considered before running the notebook.

Ensure that the `dataset_dir` in the `ml_config.yml` config file points to the path of the dataset.
It is assumed that the dataset is placed in the `prefix_dir` subdirectory (`data/` by default).

Before training a new model, make sure that there is no checkpoint under the `machine_learning_dir` subdirectory (`ML/` by default --> `ML/checkpoints/chk.pt`).
If there is a checkpoint, then the model will continue from that checkpoint instead.

Ensure to set the `output_name` in the `ml_config.yml` config file to the output filename (with extension).

Now the notebook can be executed.

When the training is finished, the model state will be output to the specified location.
The state of the model is stored in the `model_state_dict` dictionary entry when loading the model.

## Machine Learning Evaluation

The `ml_analysis.ipynb` notebook is responsible for evaluating the machine learning model on the previously created dataset.
It requires `ml_commons.py` and `ml_config.yml`.

Ensure that the `dataset_dir` in the `ml_config.yml` config file points to the path of the dataset that should be used for evaluation.
It is assumed that the dataset is placed in the `prefix_dir` subdirectory (`data/` by default).

Set the `state_filepath` to point to the previously created machine learning model state file.

Now the notebook can be executed.