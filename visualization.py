import matplotlib.pyplot as plt


def plotPredictions(yTest, yPred, fileName, r2, mse, numEstimators, seed,
                     includeCycleNum, testSize=None, trainFiles=None):
    """
    creates a scatter plot of predicted vs true capacity values.

    input:
        ytest (numpy.ndarray): actual capacity values
        ypred (numpy.ndarray): predicted capacity values
        filename (str): name of the data file being visualized
        r2 (float): r-squared score
        mse (float): mean squared error
        numestimators (int): number of trees used in the random forest
        seed (int): random seed used for training
        includecyclenum (bool): whether cycle number was included as a feature
        testsize (float, optional): proportion of data used for testing
        trainfiles (list, optional): list of files used for training (for combined mode)

    output:
        none (displays matplotlib figure)

    function:
        generates a scatter plot comparing predicted vs actual capacity values with
        model performance metrics and configuration details displayed in a text box
        at the bottom of the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(yTest, yPred, label='predicted vs true capacity')
    plt.xlabel('true capacity')
    plt.ylabel('predicted capacity')
    plt.title(f'predicted vs true capacity for {fileName}')
    plt.legend()

    plt.gcf().set_size_inches(8, 6)
    plt.subplots_adjust(bottom=0.35)

    features = 'time, voltage, current, temperature'
    if includeCycleNum:
        features += ', cycle_number'

    textLines = [
        f'r^2: {r2}',
        f'mse: {mse}',
        f'num estimators: {numEstimators}',
        f'seed: {seed}',
        f'input features: {features}'
    ]

    if testSize is not None:
        trainPct = int((1 - testSize) * 100)
        testPct = int(testSize * 100)
        textLines.insert(3, f'train/test split: {trainPct}/{testPct}')

    if trainFiles is not None:
        textLines.append(f'files used to train: {trainFiles}')

    textstr = '\n'.join(textLines)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.figtext(0.5, 0.05, textstr, ha="center", fontsize=10, bbox=props)
    plt.show()


def plotTimeSeries(time, indicesTest, yTest, yPred, fileName, testSize):
    """
    creates a time series plot of measured vs predicted capacity over time.

    input:
        time (numpy.ndarray): time values for the entire dataset
        indicestest (numpy.ndarray): indices of test data points
        ytest (numpy.ndarray): actual capacity values for test set
        ypred (numpy.ndarray): predicted capacity values for test set
        filename (str): name of the data file being visualized
        testsize (float): proportion of data used for testing

    output:
        none (displays matplotlib figure)

    function:
        plots battery capacity degradation over time, showing both measured and
        predicted values. line style and markers adapt based on test set size for
        better visualization (markers for large test sets, solid lines for small ones).
    """
    plt.figure()
    plt.plot(time[indicesTest], yTest, label='measured capacity',
             color='blue', linewidth=1 if testSize <= 0.2 else None,
             marker='o' if testSize > 0.2 else None)
    plt.plot(time[indicesTest], yPred, label='predicted capacity',
             color='red', linestyle='--', linewidth=1 if testSize <= 0.2 else None,
             marker='x' if testSize > 0.2 else None)

    plt.xlabel('time')
    plt.ylabel('capacity')
    plt.title(f'capacity over time ({fileName})')
    plt.legend()
    plt.gcf().set_size_inches(8, 6)
    plt.show()
    plt.close()
