from models.lite_intermediate import LITEIntermediate
from utils_file import training_nn_for_seeds



training_nn_for_seeds(
    # datasets= ['ArticularyWordRecognition', 'BasicMotions', 'Cricket', 'NATOPS', 'RacketSports', 'ArticularyWordRecognition', 'AtrialFibrillation', 'StandWalkJump'], 
    datasets=['daily_sports_activities', 'wear_inertial'],
    seeds= list(range(1, 11)),
    used_model = LITEIntermediate,
    is_debbug=False,
    num_ensembles=5
)