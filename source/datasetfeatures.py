# Prints out the features of one of the testing datasets
# Original code from:
# https://machinelearningmastery.com/standard-machine-learning-datasets-for-imbalanced-classification/

import argparse
import pandas as pd
import yaml
from numpy import unique

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Config file input')

    parser.add_argument('--datasetconfig', type=str,
                        default='./config/dataconfig/pimaconfig.yml',
                        help='Dataset')

    args = parser.parse_args()
    config = args.datasetconfig

    with open(config, "r") as read_file:
        dataconfig = yaml.load(read_file, Loader=yaml.FullLoader)

    if len(dataconfig['fileconfig']['dataFile']) == 1 or type(dataconfig['fileconfig']['dataFile']) == str:
        if dataconfig['dataconfig']['dataname'] == 'BANKMARKETING':
            dataframe = pd.read_csv(dataconfig['fileconfig']['dataFile'], header=dataconfig['dataconfig']['header'],sep=';')

        if dataconfig['dataconfig']['dataname'] == 'ADULT':
            dataframe = pd.read_csv(dataconfig['fileconfig']['dataFile'], header=dataconfig['dataconfig']['header'],
                                    sep=', ')

        else:
            dataframe = pd.read_csv(dataconfig['fileconfig']['dataFile'], header=dataconfig['dataconfig']['header'])
        if dataconfig['dataconfig']['featureLabels'] is not None:
            dataframe.columns = dataconfig['dataconfig']['featureLabels']

    else:

        trainframe = pd.read_csv(dataconfig['fileconfig']['dataFile'][0], header=dataconfig['dataconfig']['header'])
        testframe = pd.read_csv(dataconfig['fileconfig']['dataFile'][1], header=dataconfig['dataconfig']['header'])
        dataframe = pd.concat([trainframe, testframe], ignore_index=True)

    values = dataframe.values

    if dataconfig['dataconfig']['dataname'] == 'DOTA2':
        X, y = values[:, 1+dataconfig['dataconfig']['featOffset']:], values[:, 0]
    else:
        X,y = values[:, dataconfig['dataconfig']['featOffset']:-1], values[:, -1]

    # Remove entries with missing data from adult dataset
    if dataconfig['dataconfig']['dataname'] == 'ADULT':
        remidxs = []
        for i in range(len(X)):

            for val in X[i, :]:
                if type(val) == str:
                    if '?' == val.strip():
                        remidxs.append(i)
                        break

        keepidxs = [i for i in range(len(X)) if i not in remidxs]
        X = X[keepidxs, :]
        y = y[keepidxs]

        for i in range(len(y)):
            y[i] = y[i].strip()

        for i, label in enumerate(y):
            if label == '>50K.':
                y[i] = '>50K'
            if label == '<=50K.':
                y[i] = '<=50K'

        print("Train split perc: {:.1f}%".format(len(trainframe)/(len(trainframe) + len(testframe)) * 100))

    if dataconfig['dataconfig']['dataname'] == 'NEWS':

        for idx, val in enumerate(y):
            y[idx] = int(val) // 10000  # Finer categories
            if y[idx] >= 9:
                y[idx] = 9

    n_rows = X.shape[0]
    n_cols = X.shape[1]
    classes = unique(y)
    n_classes = len(classes)
    # summarize
    print('Features: {}'.format(dataframe.columns.values))
    print('N Examples: %d' % n_rows)
    print('N Features: %d' % n_cols)
    print('N Classes: %d' % n_classes)
    print('Classes: %s' % classes)
    print('Class Breakdown:')
    # class breakdown
    breakdown = ''
    for c in classes:
        total = len(y[y == c])
        ratio = (total / float(len(y))) * 100
        print(' - Class %s: %d (%.5f%%)' % (str(c), total, ratio))






