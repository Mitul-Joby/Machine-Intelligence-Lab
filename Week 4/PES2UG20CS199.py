'''
UE20CS302 (D Section)
Machine Intelligence
Week 4: k Nearest Neighbours

Mitul Joby
PES2UG20CS199
'''

import numpy as np

class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):
        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p


    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """
        self.data = data
        self.target = target.astype(np.int64)
        return self


    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        dist = np.sum(np.abs(x[:, np.newaxis, :] - self.data) ** self.p, axis=2) ** (1 / self.p)     
        return dist.astype(np.float64)


    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Note that the point itself is not to be included in the set of k Nearest Neighbours
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        dist = self.find_distance(x)
        neigh_dists = np.sort(dist, axis=1)[:, :self.k_neigh]
        idx_of_neigh = np.argsort(dist, axis=1)[:, :self.k_neigh]
        return neigh_dists.astype(np.float64), idx_of_neigh.astype(np.int64)


    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        neigh_dists, idx_of_neigh = self.k_neighbours(x)
        pred = np.zeros(x.shape[0])
        if self.weighted:
            for i in range(pred.shape[0]):
                pred[i] = np.bincount(self.target[idx_of_neigh[i]], weights= 1/neigh_dists[i]).argmax()
        else:
            for i in range(pred.shape[0]):
                pred[i] = np.bincount(self.target[idx_of_neigh[i]]).argmax()
        return pred.astype(np.int64)


    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        pred = self.predict(x)
        accuracy = (np.sum(pred == y) / y.shape[0]) * 100
        return accuracy.astype(np.float64)
