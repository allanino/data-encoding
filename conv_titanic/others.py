import scipy
import numpy as np
import operator
import csv
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import random

# loading csv data into numpy array
def read_data(f, header=True, test=False, rows=0):
    data = []
    labels = []

    csv_reader = csv.reader(open(f, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index = index + 1
        if rows > 0 & index > rows:
            break
        if header and index == 1:
            continue

        if not test:
            labels.append(float(row[len(row) -1]))
            row = row[0:len(row) -1]

        data.append(np.array(np.float64(row)))
    return (data, labels)

def predictGNB(train, labels, test):
    gnb = GaussianNB()
    gnb.fit(train, labels)
    gnb_predictions = gnb.predict(test)
    return gnb_predictions

def predictRF(train, labels, test):
    rf = RandomForestClassifier(n_estimators=200, n_jobs=3)
    rf.fit(train, labels)
    rf_predictions = rf.predict(test)
    return rf_predictions

def predictSVC(train, labels, test):
    clf = SVC()
    clf.fit(train, labels)
    svc_predictions = clf.predict(test)
    return svc_predictions

def predictKNN(train,labels,test):
    knn = neighbors.KNeighborsClassifier(weights='distance')
    knn.fit(train, labels) 
    predictions = knn.predict(test)
    return predictions

if __name__ == '__main__':

    # GNB (Gaussian Naive Bayes)
    accuracy_list = []
    for i in range(0,50):       
        # Read the data
        data, all_label = read_data("titanic.csv", header=False)

        # Random shuffle the data
        myorder = [int(len(data)*random.random()) for i in xrange(len(data))]
        data = [ data[i] for i in myorder]
        all_label = [all_label[i] for i in myorder]

        # Split training and test in 80%/20%
        train_size = int(len(data)*0.8)
        test_size = len(data) - train_size

        train = data[0:train_size]
        labels = all_label[0:train_size] 

        test = data[train_size:]
        tmlp = all_label[train_size:] 

        # Train the classifier and make predictions
        predictions = predictGNB(train,labels, test)

        # Compute accuracy
        accuracy_list.append(accuracy_score(tmlp, predictions))
    
    print "GNB:",np.mean(accuracy_list)

    # RF (Random Forests)
    accuracy_list = []
    for i in range(0,50):       
        # Read the data
        data, all_label = read_data("titanic.csv", header=False)

        # Random shuffle the data
        myorder = [int(len(data)*random.random()) for i in xrange(len(data))]
        data = [ data[i] for i in myorder]
        all_label = [all_label[i] for i in myorder]

        # Split training and test in 80%/20%
        train_size = int(len(data)*0.8)
        test_size = len(data) - train_size

        train = data[0:train_size]
        labels = all_label[0:train_size] 

        test = data[train_size:]
        tmlp = all_label[train_size:] 

        # Train the classifier and make predictions
        predictions = predictRF(train,labels, test)

        # Compute accuracy
        accuracy_list.append(accuracy_score(tmlp, predictions))
    
    print "RF:",np.mean(accuracy_list)

    # SVM (Support Vector Machine)
    accuracy_list = []
    for i in range(0,50):       
        # Read the data
        data, all_label = read_data("titanic.csv", header=False)

        # Random shuffle the data
        myorder = [int(len(data)*random.random()) for i in xrange(len(data))]
        data = [ data[i] for i in myorder]
        all_label = [all_label[i] for i in myorder]

        # Split training and test in 80%/20%
        train_size = int(len(data)*0.8)
        test_size = len(data) - train_size

        train = data[0:train_size]
        labels = all_label[0:train_size] 

        test = data[train_size:]
        tmlp = all_label[train_size:] 

        # Train the classifier and make predictions
        predictions = predictSVC(train,labels, test)

        # Compute accuracy
        accuracy_list.append(accuracy_score(tmlp, predictions))
    
    print "SVM:",np.mean(accuracy_list)


    # KNN (k-Nearest Neighbors)
    accuracy_list = []
    for i in range(0,50):       
        # Read the data
        data, all_label = read_data("titanic.csv", header=False)

        # Random shuffle the data
        myorder = [int(len(data)*random.random()) for i in xrange(len(data))]
        data = [ data[i] for i in myorder]
        all_label = [all_label[i] for i in myorder]

        # Split training and test in 80%/20%
        train_size = int(len(data)*0.8)
        test_size = len(data) - train_size

        train = data[0:train_size]
        labels = all_label[0:train_size] 

        test = data[train_size:]
        tmlp = all_label[train_size:] 

        # Train the classifier and make predictions
        predictions = predictKNN(train,labels, test)

        # Compute accuracy
        accuracy_list.append(accuracy_score(tmlp, predictions))
    
    print "KNN:",np.mean(accuracy_list)