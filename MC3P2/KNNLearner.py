"""
A simple wrapper for KNN. 
"""

import numpy as np


class KNNLearner(object):
    def __init__(self, k, verbose = False):
        self.k = k
        pass # move along, these aren't the drones you're looking for

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.Xtrain = dataX
        self.Ytrain = dataY

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        yPred = np.zeros(points.shape[0])
        i=0
        for point in points:
            pointCopies = (np.ones(shape=(self.Xtrain.shape[0],self.Xtrain.shape[1])) * point)
            euclid = np.sqrt(((self.Xtrain-pointCopies)**2).sum(axis=1))
            sorted = np.argsort(euclid)
            yPred[i] = np.mean(self.Ytrain[sorted[:self.k]])
            i+=1
        return yPred