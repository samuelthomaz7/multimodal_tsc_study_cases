

from models.fully_convolutional_network_1d_intermediateGAP import FullyConvolutionalNetwork1DIntermediateGAP
from utils_file import training_nn_for_seeds


training_nn_for_seeds(
    datasets= ['daily_sports_activities', 'wear_inertial'], 
    seeds= list(range(1, 11)),
    used_model = FullyConvolutionalNetwork1DIntermediateGAP
)