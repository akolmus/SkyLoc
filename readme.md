# Skylocalization of Gravitational Waves
A 1D convolutional neural network parametrizes either a Von Mises-Fisher distribution or a Kent distribution for each given strain of data. The Kent distribution is currently not working as intended...

## How to run
One needs to make signal and noise sequences using the scripts create_data.py and create_noise.py, the settings are specified in the similarily named yaml files. Once the data has been created, the create_model.py can create/train a model.