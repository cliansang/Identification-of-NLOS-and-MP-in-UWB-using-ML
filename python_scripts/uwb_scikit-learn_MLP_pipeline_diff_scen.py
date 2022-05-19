# Authors:	Cung Lian Sang, Bastian Steinhagen, Michael Adams
# Emails:	{csang, bsteinhagen, madams}@techfak.uni-bielefeld.de
# Cognitronics and Sensor Systems Group, CITEC, Bielefeld University

# Import pandas and numpy for data manipulation
import pandas as pd
import numpy as np
import time

# Import modules from Scikit-learn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split   # Import train_test_split function
from sklearn import metrics   # import metrics modules for accuracy calculation
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import dump, load                      # For saving and loading the trained model


# Create a pipeline object for our model
pipe_MLP = make_pipeline(StandardScaler(),
                         MLPClassifier(solver='adam',
                                       hidden_layer_sizes=(100,100,100,), # 3 hidden layers with (100x100x100) neurons
                                       random_state=0,
                                       max_iter=500,           # TODO: tune it later
                                       verbose=True
                                       )
                         )

# MLP without preprocessing phase
# pipe_MLP = make_pipeline(MLPClassifier(solver='adam',
#                                        hidden_layer_sizes=(100,100,100,), # 3 hidden layers with (100x100x100) neurons
#                                        random_state=0,
#                                        max_iter=500,           # TODO: tune it later
#                                        verbose=True
#                                        )
#                          )

print("Data Pipeline (StandardScaler >> MLPClassifier) is created!\n")
print(pipe_MLP)

# Load the measured data in CSV format using pandas
#uwb_raw = "../uwb_raw_data/uwbFullDataSet.txt"
uwb_raw = "../uwb_raw_data/downSample_uwb_FullDataSet.txt"
#uwb_raw = "../uwb_raw_data/upsample_uwb_FullDataSet.txt"
data = pd.read_csv(uwb_raw)
#print(data.shape)
#print(data)
data.head()

# Extracted features 
#X = data[['Dist', 'FP', 'FPAmp1', 'FPAmp2', 'FPAmp3', 'CIR', 'stdN', 'maxN', 'preamCnt', 'RSS_CIR', 'RSS_FP', 'powerDiff']]
#X = data[['Dist', 'FP', 'FPAmp1', 'FPAmp2', 'FPAmp3', 'CIR', 'stdN', 'maxN', 'preamCnt', 'RSS_CIR']]
#X = data[['Dist', 'FPAmp1', 'FPAmp2', 'FPAmp3', 'CIR']]  # Selected Features
X = data[['Dist', 'FPAmp1', 'CIR']]  # Only 3 Features
#X = data[['Dist', 'FPAmp2', 'CIR']]  # Only 3 Features
#X = data[['Dist', 'FP', 'CIR']]  # Only 3 Features
#X = data[['FPAmp1', 'CIR']]  # Only 2 Features
#X = data[['Dist', 'FPAmp1']]  # Only 2 Features
#X = data[['Dist', 'CIR']]  # Only 2 Features

y = data['Label']  # Labels
print("Selected features among 13 available in total: \n", X)
print("Class labels:\n", y)

###############  Test data set collected at different scenarios (rooms, hall, etc )  from training #################
# Load the Test Data set for different scenarios
uwb_test_raw = "../uwb_raw_test_dataset_diff_scen/uwbTestDataSetDiffScenarios_balanced.txt"
test_data = pd.read_csv(uwb_test_raw)
test_data.head()

# Test data only for several scenarios which is different from the collected data at Training conditions
# (rooms, corridors etc.) mentioned above
#X_test_diffScen = test_data[['Dist', 'FP', 'FPAmp1', 'FPAmp2', 'FPAmp3', 'CIR', 'stdN', 'maxN', 'preamCnt', 'RSS_CIR', 'RSS_FP', 'powerDiff']] # 12 features
X_test_diffScen = test_data[['Dist', 'FPAmp1', 'CIR']]    # 3 features
y_test_diffScen = test_data['Label']

#########################################

# create a python lists to store data (initialization)
acc_MLP = []
acc_MLP_test_diffScen = []
trTime_MLP = []
tesTime_MLP = []
pred_10iter = []
true_10iter = []
pred_10iter_test_diffScen = [] # only one true involved in this case. can be added at the last column

for x in np.arange(10):
    # Split dataset into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3) # 70% training and 30% test
    # tr_dataSize = np.shape(X_train)   # save just the last data since sizes are always the same
    # test_dataSize = np.shape(X_test)

    print("\nTraining in progress! This may take some times ... \n")
    t0 = time.time()
    # Train the whole pipeline
    pipe_MLP.fit(X_train, y_train)
    trTime = time.time() - t0
    # print("Training Time: %0.3fs" % (time.time() - t0))
    trTime_MLP.append(trTime)

    # Predict the whole pipeline
    t1 = time.time()
    y_pred = pipe_MLP.predict(X_test)
    tesTime = time.time() - t1
    # print("Testing Time: %0.3fs" % (time.time() - t1))
    tesTime_MLP.append(tesTime)

    # Predict Real test Data using the whole Pipeline
    y_pred_diffScen = pipe_MLP.predict(X_test_diffScen)

    # prediction and true labels for each iteration
    pred_10iter.append(y_pred)
    true_10iter.append(y_test)
    pred_10iter_test_diffScen.append(y_pred_diffScen)

    # Print some results
    print("\nTraining Time in iter %d: %0.2fs" % (x, trTime))
    print("\nTesting Time in iter %d: %0.2fs" % (x, tesTime))
    # Model accuracy, how often is the classifier correct?
    print("\nTest Accuracy in MLP at iter %d: %0.4f" % (x, metrics.accuracy_score(y_test, y_pred)))
    print("\nTest Accuracy in MLP at diff scenario in iter %d: %0.4f" % (x, metrics.accuracy_score(y_test_diffScen, y_pred_diffScen)))
    acc_MLP.append(metrics.accuracy_score(y_test, y_pred))
    acc_MLP_test_diffScen.append(metrics.accuracy_score(y_test_diffScen, y_pred_diffScen))

# Finished training for 10 iterations
print("\n\n\nTraining and Testing of 10 iterations finished!\n")

# append only once the true class for test Dataset which was collected in different scenarios
pred_10iter_test_diffScen.append(y_test_diffScen)

# Tranpose the rows into columns (appended by each iterations) and stacked them togethers as features (column results)
MLP_pred_true_10iter = np.column_stack((np.array(pred_10iter).T, np.array(true_10iter).T))
print("Prediction results save as CSV in file name 'MLP_pred_true_10iter.txt'\n")
np.savetxt("MLP_pred_true_10iter_%s.txt" %time.strftime("%d%m%Y_%H%M%S"), MLP_pred_true_10iter, delimiter=',',
           header="Prediction[0:9]th iteration,  True[0:9]th iteration", comments="")

# save the predict-true pair for the real test dataset at different scenarios
MLP_pred_true_10iter_test_diffScen = np.array(pred_10iter_test_diffScen).T
np.savetxt("MLP_pred_true_10iter_testDiffScen_%s.txt" %time.strftime("%d%m%Y_%H%M%S"), MLP_pred_true_10iter_test_diffScen,
           delimiter=',', header="Prediction[0:9],  True_Test", comments="")

# Print some overall results
print("\nTotal Data Size, Training size, Test Size:")
print(np.shape(X)), print(np.shape(X_train)), print(np.shape(X_test))

print("\nmean accuracy in MLP: %0.4f +/- %f" % (np.mean(acc_MLP), np.std(acc_MLP)))
print("\nmean accuracy in MLP at diff scenario: %0.4f +/- %f" % (np.mean(acc_MLP_test_diffScen), np.std(acc_MLP_test_diffScen)))
MLP_acc_tr_tes = np.column_stack((acc_MLP, trTime_MLP, tesTime_MLP))       # combine numpy vectors as a column stack
print("\nmean Training Time with std in MLP: %0.2f +/- %f (sec)" % (np.mean(trTime_MLP), np.std(trTime_MLP)))
print("\nmean Testing Time with std in MLP: %0.2f +/- %f (sec)" % (np.mean(tesTime_MLP), np.std(tesTime_MLP)))

print("\nresults save in the file name 'MLP_10iter.txt'\n")
np.savetxt("MLP_10iter_%s.txt" %time.strftime("%d%m%Y_%H%M%S"), MLP_acc_tr_tes, delimiter=',',
           header="Accuracy, Training_Time, Test_Time", comments="")
print(MLP_acc_tr_tes)
