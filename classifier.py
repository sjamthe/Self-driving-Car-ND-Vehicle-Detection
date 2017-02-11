import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import sys
#import fnmatch
import time
import random
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.externals import joblib

def trainLinearSVC(X_train, y_train):
    """
    Best score: 0.802
    Best parameters set:
        C: 1000
        max_iter: 100
        penalty: 'l2'
        for LogisticRegression
    parameters = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l1', 'l2'],
         'max_iter': [100]
    }
    """
    parameters = {
        'C': [.01],
        'penalty': ['l2'],
         'max_iter': [100]
    }

    # Use a linear SVC
    print("Building logistic model")
    svc = LogisticRegression()
    # Check the training time for the SVC
    grid_search = GridSearchCV(svc, parameters,n_jobs=1)
    grid_search.fit(X_train, y_train)

    print ("Best score: %0.3f" % (grid_search.best_score_))

    print ("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print ("\t%s: %r" % (param_name, best_parameters[param_name]))

    return grid_search

def trainSGDClassifier(X_train, y_train):
    """
    Best score: 0.791
    Best parameters set:
        alpha: 0.001
        loss: 'log'
        n_iter: 50
        penalty: 'elasticnet'

    parameters = {
        'loss': ('log', 'hinge','modified_huber'),
        'penalty': ['l1', 'l2', 'elasticnet'],
        'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001],
         'n_iter': [50]
    }
    """
    parameters = {
        'loss': ['log'],
        'penalty': ['elasticnet'],
        'alpha': [0.001],
         'n_iter': [50]
    }
    # Use a linear SVC
    print("Building SGDClassifier model")
    svc = SGDClassifier()
    # Check the training time for the SVC
    grid_search = GridSearchCV(svc, parameters)
    grid_search.fit(X_train, y_train)

    print ("Best score: %0.3f" % (grid_search.best_score_))
    print ("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print ("\t%s: %r" % (param_name, best_parameters[param_name]))

    return grid_search

def train_classifier(X, y, modelfile):

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))
    t=time.time()
    if 'svc' in modelfile:
        svc = trainLinearSVC(X_train, y_train)
    else:
        svc = trainSGDClassifier(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print(svc.predict_proba(X_test[0:n_predict]))
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For  ',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    #Save the model to a pickle file
    joblib.dump((X_scaler,svc), modelfile+'.pkl')

"""
Program helps try out various model hyper parameters and features
argv[1] is the model filename to save model to
"""
def main():
    modelfile = sys.argv[1]
    car_features = None
    noncar_features = None
    # Read in car and non-car features from pickled files
    if 'v1' in modelfile:
        version = 'v1'
    else:
        version = 'v2'

    featurefiles = [
             '../data/KITTI-cars-features-'+version+'.pkl'
             ,'../data/KITTI-non-cars-features-'+version+'.pkl'
             ,'../data/GTI-cars-features-'+version+'.pkl'
             ,'../data/GTI-non-cars-features-'+version+'.pkl'
             #,'../data/Udacity-cars5-features-'+version+'.pkl'
             #,'../data/Udacity-non-cars5-features-'+version+'.pkl'
             #,'../data/Udacity-cars10-features-'+version+'.pkl'
             #,'../data/Udacity-non-cars10-features-'+version+'.pkl'
             #,'../data/Udacity-cars11-features-'+version+'.pkl'
             #,'../data/Udacity-non-cars11-features-'+version+'.pkl'
             #,'../data/Udacity-cars15-features-'+version+'.pkl'
             #,'../data/Udacity-non-cars15-features-'+version+'.pkl'
             ,'../data/Project-non-cars-features-'+version+'.pkl'
             ]

    for file in featurefiles:
        features = joblib.load(file)
        print ("Read %d features from %s" % (len(features), file))
        if 'non-cars' in file:
            if noncar_features is None:
                noncar_features = np.copy(features)
            else:
                noncar_features = np.concatenate((noncar_features, features))
            #print("noncar_features",noncar_features.shape)
        else:
            if car_features is None:
                car_features = np.copy(features)
            else:
                car_features = np.concatenate((car_features, features))
            #print("car_features",car_features.shape)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, noncar_features)).astype(np.float64)
    #print("total features", X.shape)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

    print("Training classifier on vehicles", len(car_features), "non-vehicles",
            len(noncar_features))
    train_classifier(X, y, modelfile)

""" Main function for testing purpose. Input sequence of images """
if __name__ == '__main__':
    main()
