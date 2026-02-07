import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from visualization import plotPredictions, plotTimeSeries


def loadData(filePath, includeCycleNum=True):
    """
    loads and prepares battery data from a csv file.

    input:
        filepath (str): name of the csv file in the data directory
        includecyclenum (bool): whether to include cycle number as a feature (default: true)

    output:
        x (numpy.ndarray): feature matrix with shape (n_samples, n_features)
        capacity (numpy.ndarray): target capacity values with shape (n_samples, 1)
        time (numpy.ndarray): time values with shape (n_samples, 1)

    function:
        reads battery cycling data from csv, extracts time, voltage, current, temperature,
        and capacity columns. optionally includes cycle number as an additional feature.
        returns preprocessed feature matrix and target values.
    """
    data = pd.read_csv("data/" + filePath)

    # extract relevant columns
    # time_s: time since beginning of experiment in seconds
    time = data.iloc[:, 0].values.reshape(-1, 1)

    # ecell_v: cell voltage
    voltage = data.iloc[:, 1].values.reshape(-1, 1)

    # i_ma: cell current in milliamperes
    current = data.iloc[:, 2].values.reshape(-1, 1)

    # temperature__c: cell surface temperature in degrees celsius
    temperature = data.iloc[:, 7].values.reshape(-1, 1)

    # qdischarge_ma_h: charge extracted from cell during discharge in milliampere-hours
    capacity = data.iloc[:, 6].values.reshape(-1, 1)

    if includeCycleNum:
        # cyclenumber: cycle number as recorded by the cell tester
        cycleNumber = data.iloc[:, 8].values.reshape(-1, 1)
        X = np.concatenate((time, voltage, current, temperature, cycleNumber), axis=1)
    else:
        X = np.concatenate((time, voltage, current, temperature), axis=1)

    return X, capacity, time


def trainIndividualFiles(filePaths, includeCycleNum, testSize, numEstimators, seed):
    """
    trains random forest models on individual battery files with train/test split.

    input:
        filepaths (list): list of csv file names to process
        includecyclenum (bool): whether to include cycle number as a feature
        testsize (float): proportion of data to use for testing (0.0 to 1.0)
        numestimators (int): number of trees in the random forest
        seed (int): random seed for reproducibility

    output:
        none (prints metrics and displays plots for each file)

    function:
        for each file, loads data, splits into train and test sets, trains a random forest
        regressor, makes predictions, calculates mse and r^2 metrics, and generates
        visualization plots showing prediction accuracy and time series degradation.
    """
    for filePath in tqdm(filePaths, desc="processing files"):
        # load data
        X, y, time = loadData(filePath, includeCycleNum)

        # split the data into training and testing sets
        indices = np.arange(len(y))
        xTrain, xTest, yTrain, yTest, indicesTrain, indicesTest = train_test_split(
            X, y, indices, test_size=testSize, random_state=seed
        )

        # create and train the model
        rfRegressor = RandomForestRegressor(n_estimators=numEstimators, random_state=seed)
        rfRegressor.fit(xTrain, yTrain.ravel())

        # make predictions
        yPred = rfRegressor.predict(xTest)

        # get evaluation metrics
        mse = mean_squared_error(yTest, yPred)
        r2 = r2_score(yTest, yPred)
        print(f'for {filePath}:')
        print(f'mse: {mse}')
        print(f'r^2 score: {r2}')

        # plot results
        plotPredictions(yTest, yPred, filePath, r2, mse, numEstimators,
                        seed, includeCycleNum, testSize=testSize)
        plotTimeSeries(time, indicesTest, yTest, yPred, filePath, testSize)


def trainCombinedFiles(filePaths, testFile, includeCycleNum, numEstimators, seed):
    """
    trains a random forest model on combined battery files and tests on a held-out file.

    input:
        filepaths (list): list of all csv file names available
        testfile (str): name of the file to use for testing (excluded from training)
        includecyclenum (bool): whether to include cycle number as a feature
        numestimators (int): number of trees in the random forest
        seed (int): random seed for reproducibility

    output:
        none (prints metrics and displays plots)

    function:
        combines data from all training files into a single dataset, trains a random forest
        model on the combined data, evaluates performance on the held-out test file,
        calculates mse and r^2 metrics, and generates visualization plots. tests
        cross-battery generalization capability.
    """
    trainFiles = [f for f in filePaths if f != testFile]

    # determine number of features
    numFeatures = 5 if includeCycleNum else 4

    # placeholder for combined training data
    combinedXTrain = np.empty((0, numFeatures))
    combinedYTrain = np.empty((0, 1))

    # combine the training data from multiple files
    for filePath in tqdm(trainFiles, desc="processing files"):
        X, y, _ = loadData(filePath, includeCycleNum)
        combinedXTrain = np.vstack((combinedXTrain, X))
        combinedYTrain = np.vstack((combinedYTrain, y))

    # train the model using the combined training data
    rfRegressor = RandomForestRegressor(n_estimators=numEstimators, random_state=seed)
    rfRegressor.fit(combinedXTrain, combinedYTrain.ravel())

    # read the test data
    xTest, yTest, _ = loadData(testFile, includeCycleNum)

    # make predictions on the test set
    yPred = rfRegressor.predict(xTest)

    # evaluate the model's performance on the test set
    mse = mean_squared_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)
    print(f'test results for {testFile}:')
    print(f'mse: {mse}')
    print(f'r^2 score: {r2}')

    # plot results
    plotPredictions(yTest, yPred, testFile, r2, mse, numEstimators,
                    seed, includeCycleNum, trainFiles=trainFiles)
