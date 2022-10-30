'''
UE20CS302 (D Section)
Machine Intelligence
Week 6 - Support Vector Machines

Mitul Joby
PES2UG20CS199
'''

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
import pandas as pd
import numpy as np


class SVM:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        data = pd.read_csv(self.dataset_path)
        self.X = data.iloc[:, 0:-1]
        self.y = data.iloc[:, -1]

    def solve(self):
        """
        Build an SVM model and fit on the training data
        The data has already been loaded in from the dataset_path

        Refrain to using SVC only (with any kernel of your choice)

        You are free to use any any pre-processing you wish to use
        Note: Use sklearn Pipeline to add the pre-processing as a step in the model pipeline
        Refrain to using sklearn Pipeline only not any other custom Pipeline if you are adding preprocessing

        Returns:
            Return the model itself or the pipeline(if using preprocessing)
        """

        svc = SVC(C = 2.0)
        scaler = StandardScaler()  # 90.81%
        # scaler = MinMaxScaler()  # 88.44%
        # scaler = MaxAbsScaler()  # 90.07%
        normalizer = Normalizer()
        pipe = Pipeline( [('scaler', scaler), ('normalizer', normalizer), ('svc', svc)])
        pipe.fit(self.X, self.y)
        return pipe
