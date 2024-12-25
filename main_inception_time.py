from models.inception_time import InceptionTime
from utils_file import training_nn_for_seeds



training_nn_for_seeds(
    # datasets= ['ArticularyWordRecognition', 'BasicMotions', 'Cricket', 'NATOPS', 'RacketSports', 'ArticularyWordRecognition', 'AtrialFibrillation', 'StandWalkJump'], 
    datasets= ['daily_sports_activities', 'wear_inertial'],
    seeds= list(range(1, 11)),
    used_model = InceptionTime,
    is_debbug=False,
    num_ensembles=5
)