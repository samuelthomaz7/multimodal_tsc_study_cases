import numpy as np
import os
import pickle


daily_sports_activities = {
    'X': np.load('/home/stbastos/experiments/health_tests/multimodal_tsc_study_cases/downloaded_datasets/daily-and-sports-activities/daily-and-sports-activities/x_data.npy'),
    'y': np.load('/home/stbastos/experiments/health_tests/multimodal_tsc_study_cases/downloaded_datasets/daily-and-sports-activities/daily-and-sports-activities/y_data.npy'),
    'metadata': np.load('/home/stbastos/experiments/health_tests/multimodal_tsc_study_cases/downloaded_datasets/daily-and-sports-activities/daily-and-sports-activities/metadata.npy', allow_pickle=True).item()
}

daily_sports_activities['metadata']['problemname'] = 'daily_and_sports_activities'

wear_inertial_raw = {
    'X_train': np.load('/home/stbastos/experiments/health_tests/multimodal_tsc_study_cases/downloaded_datasets/wear-inertial-1s-50hz/wear-inertial/train_data_x.npy', allow_pickle=True),
    'X_test': np.load('/home/stbastos/experiments/health_tests/multimodal_tsc_study_cases/downloaded_datasets/wear-inertial-1s-50hz/wear-inertial/test_data_x.npy', allow_pickle=True),
    'y_train': np.load('/home/stbastos/experiments/health_tests/multimodal_tsc_study_cases/downloaded_datasets/wear-inertial-1s-50hz/wear-inertial/train_data_y.npy', allow_pickle=True),
    'y_test': np.load('/home/stbastos/experiments/health_tests/multimodal_tsc_study_cases/downloaded_datasets/wear-inertial-1s-50hz/wear-inertial/test_data_y.npy', allow_pickle=True),
    'train_metadata': np.load('/home/stbastos/experiments/health_tests/multimodal_tsc_study_cases/downloaded_datasets/wear-inertial-1s-50hz/wear-inertial/train_metadata.npy', allow_pickle=True).item(),
    'test_metadata': np.load('/home/stbastos/experiments/health_tests/multimodal_tsc_study_cases/downloaded_datasets/wear-inertial-1s-50hz/wear-inertial/test_metadata.npy', allow_pickle=True).item()
}

wear_inertial = {
    'X': np.concatenate((wear_inertial_raw['X_train'], wear_inertial_raw['X_test']), axis = 0),
    'y': np.concatenate((wear_inertial_raw['y_train'], wear_inertial_raw['y_test']), axis = 0),
    'metadata': {
        'problemname': 'wear_inertial',
        'folds': np.concatenate((wear_inertial_raw['train_metadata']['folds'], wear_inertial_raw['test_metadata']['folds']), axis = 0)
    }
}

wear_inertial['metadata']['class_values'] = np.unique(wear_inertial['y'])
daily_sports_activities['metadata']['class_values'] = np.unique(daily_sports_activities['y'])


with open('/home/stbastos/experiments/health_tests/multimodal_tsc_study_cases/downloaded_datasets/daily_sports_activities.pkl','wb') as f:
    pickle.dump(daily_sports_activities, f)

with open('/home/stbastos/experiments/health_tests/multimodal_tsc_study_cases/downloaded_datasets/wear_inertial.pkl','wb') as f:
    pickle.dump(wear_inertial, f)

