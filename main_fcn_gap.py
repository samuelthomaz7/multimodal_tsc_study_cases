

from models.fully_convolutional_network_1dGAP import FullyConvolutionalNetwork1DGAP
from utils_file import training_nn_for_seeds


training_nn_for_seeds(
    datasets= ['wesad'], 
    seeds= list(range(1, 11)),
    used_model = FullyConvolutionalNetwork1DGAP
)