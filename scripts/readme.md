Data generation to train neural networks for the MEB Model
==========================================================

These scripts are intended to generate data to train neural networks for the 
MEB model.
These scripts should be executed in its numbered order.
Common arguments are  the data path and number of workers that should be 
used to generate the data.
Only `01_generate_initial_ensemble.py` is stochastic and depends on specific 
random generators.
To reproduce the datasets for training, evaluation and testing from the paper,
the following seeds were used:

 - `training_seed`: `42`
 - `evaluation_seed`: `10`
 - `testing_seed`: `0`

The size of the ensemble members and, thus, of the available samples is as 
follows:

 - `training_size`: `100`
 - `evaluation_size`: `20`
 - `testing_size`: `50`

To run the scripts a compiled version of the Maxwell-Elasto-Brittle model 
**has to be linked** to the `meb_model` directory.

All datasets will be generated in the specified data directory as zarr folders.
The following concisely describes the purpose of each script to run the meb 
model and to generate the raw model data:

 1. `01_generate_initial_ensemble.py`: Generates the initial ensemble of 
    members. For each ensemble member, a set of forcing parameters and a 
    random initital cohesion field are generated.
 2. `02_generate_nature_spinup.py`: Generates the spin up of the truth. To save
    space and time, only each hour is stored. The last state is used to as 
    initial conditions to generate the truth trajectories.
 3. `03_generate_nature_forecast.py`: Generates the high-resolution 
    forecast data in the same way as used for the low-resolution forecast. 
    Because the typical time-step of the high-resolution run is eight 
    seconds, only every second step is outputted. This script reuses the 
    generated forcing parameters.
 4. `04_project_to_coarse.py`: Projects the high-resolution nature run 
    trajectories and forecast data to the coarse grid. The projected 
    trajectories are used as initial conditions for the low-resolution 
    forecasts whereas the projected forecasts are the reference for the 
    training of the neural network. Currently, the runs are projected with 
    the interpolation as it is used for finite-element models.
 5. `05_generate_lr_forecast.py`: Generates the low-resolution forecast 
    data based on the projected initial conditons and the genereated forcing 
    parameters.

This raw model is then converted with the `dataset_*_*.py` into different
training datasets, e.g. for different types of input.

The `estimate_proj_weights.py` has been used to estimate the projection weights
for different Cartesian space sizes.
