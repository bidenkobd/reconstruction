**Public release of software accompanying the paper published at https://arxiv.org/abs/TBD**

It is organized as follows:

 - Generation of the training, validation, and test sets is done with the `field_generation_used_in_paper.py` script.
 - Simulation parameters are saved in `.csv` files.
 - Data preprocessing is done with `preprocessing.ipynb`.
 - The architecture of the neural network used in this work is defined in `UNET.py`, originally published in https://arxiv.org/abs/2305.07018.
 - The neural network training was performed using the `training.py` script.
 - Trained weights are saved in the `model_weights` folder.
 - Plotting of results presented in the results section of the paper is done with `plots.ipynb` (note that the test set is not provided and should be generated with the corresponding script).
 - Figures used in the paper are saved in the `fig` folder.

The pipeline is tested with the following versions of modules:

 - tensorflow v2.11.0
 - tensorflow_addons v0.23.0
 - py21cmfast v3.3.1
