from sklearn.model_selection import train_test_split
import numpy as np


class TrainTestSplit():

    def __init__(self, X_raw, y_raw, metadata = None, random_state = 42):
        self.X_raw = X_raw
        self.y_raw = y_raw
        self.metadata = metadata
        self.step_name = 'TrainTestSplit'
        self.random_state = random_state

    
    def fit(self):
        pass

    def transform(self, custom_groups = False):

        if custom_groups:
            unique_values = np.unique(ar = self.metadata['folds'])
            train_groups_size = int(0.8*len(unique_values))
            train_groups = np.random.choice(unique_values, size=train_groups_size, replace=False)
            test_groups = np.array([i for i in unique_values if i not in train_groups])

            train_mask = [i in train_groups for i in self.metadata['folds']]
            test_mask = [i in test_groups for i in self.metadata['folds']]

            self.X_train, self.X_test, self.y_train, self.y_test = self.X_raw[train_mask, :, :], self.X_raw[test_mask, :, :], self.y_raw[train_mask, :], self.y_raw[test_mask, :]


        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_raw, self.y_raw, random_state=self.random_state, train_size= 0.8)
        
        return self.X_train, self.X_test, self.y_train, self.y_test