import sys
import time
import numpy
import pandas

import sklearn.linear_model
from sklearn.decomposition import PCA

train = pandas.read_csv('data/train.csv')
test = pandas.read_csv('data/test.csv')


def main():

    print(type(test))
    start_time = time.time()
    training_set()
    end_time = time.time()

    print('execution time ',end_time - start_time)

    return 'done'


def training_set():
    print 'training data'

    n_test_samples = -1000
    n_training_samples = -10000 - n_test_samples

    train_sm = train[n_training_samples : n_test_samples]
    test_sm = train[n_test_samples : ]
    print(train_sm)

    # Initialise logistic regression
    LogReg = sklearn.linear_model.LogisticRegression()
    
    train_data = train_sm.ix[:,1:]
    train_target = train_sm.ix[:,:1]

    # Fit using (data, target)
    LogReg.fit(train_data, train_target)

    test_data = test_sm.ix[:, 1:]
    test_target = test_sm.ix[:, :1]

    # Predict fit of output
    print('predict on training data')    
    logreg_training = LogReg.predict(test_data)
    print(logreg_training)
    print(type(logreg_training))

    print('target')
    print(test_target)

    test_target_ = numpy.transpose(numpy.array(test_target))[0]
    print(test_target_)
    acc = find_accuracy(logreg_training, test_target_)
    print(acc)



    # Using PCA
    train_data_pca = dimensionality_reduction(train_sm, abs(n_test_samples))
    test_data_pca = dimensionality_reduction(test_sm, abs(n_test_samples))

    print('test_pca')
    print(train_data_pca)


    # Initialise logistic regression
    LogReg2 = sklearn.linear_model.LogisticRegression()
    
    train_data = train_data_pca[:,1:]
    train_target = train_sm.ix[:,:1]

    # Fit using (data, target)
    LogReg2.fit(train_data, train_target)

    test_target = test_sm.ix[:, :1]

    # Predict fit of output
    print('predict on training data')
    print(test_data_pca.shape)
    logreg_training = LogReg2.predict(test_data_pca[:,1:])
    print(logreg_training)
    print(type(logreg_training))

    print('target')
    print(test_target)

    test_target_ = numpy.transpose(numpy.array(test_target))[0]
    print(test_target_)
    acc = find_accuracy(logreg_training, test_target_)
    print(acc)


    return logreg_training
    

def find_accuracy(logreg_prediction, target):
    error = numpy.mean(logreg_prediction != target)
    perc_acc = 100 * (1 - error)
    return perc_acc


def dimensionality_reduction(data, component_num):
    # Use PCA (Principal Component Analysis) to cut down columns

    pca = PCA(n_components = component_num)
    pca.fit(data)
    data = pca.transform(data)
    return data


def testing_set():
    print 'test data'
    test_sm = test[-10:]
    print(test_sm)
    return test_sm


if __name__ == '__main__':
    sys.exit(main())
