#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score



# Loading and transposing the dataset

df = pd.read_csv("Train_call.csv")
df = df.T
x = df.iloc[4:,:]

# Loading the labels
labels = pd.read_csv("Train_clinical.csv")
y = labels.iloc[1:,1]


## Implementing random search for hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', None]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid space
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


## BELOW TAKES FOREVER TO RUN. DO NOT RUN IT UNLESS IT'S NECESSARY.

## First Nested cross-validation process

# Outer cross-validation
cv_outer = KFold(n_splits=5, shuffle=True, random_state=2)

# Dataframe for the inner cross-validation results
cols = {"n_estimators":[],"min_samples_split":[],"min_samples_leaf":[],"max_features":[],"max_depth":[],"bootstrap":[]}
best_parameters = pd.DataFrame(data=cols)

# Loop 10 times with 5x4 nested cross-validation
for i in range(10):
    print("In progress...{}/10".format(i+1))
    for train_ix, test_ix in cv_outer.split(x):
        # Train and test split in the inner cross-validation set
        x_in_train, x_in_test = x.iloc[train_ix, :], x.iloc[test_ix, :]
        y_in_train, y_in_test = y.iloc[train_ix], y.iloc[test_ix]

        # Configure the inner cross-validation procedure
        cv_inner = KFold(n_splits=4, shuffle=True, random_state=i)

        # Define the model
        model = RandomForestClassifier(random_state=i)

        # Define search space, use the random grid assigned above
        search = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 50, cv = cv_inner, random_state=i, n_jobs = -1)
        result = search.fit(x_in_train, y_in_train)
        best_model = result.best_estimator_

        # Evaluate model on the hold out dataset
        yhat = best_model.predict(x_in_test)

        # Evaluate the model
        acc = accuracy_score(y_in_test, yhat)

        # Store the result into the dataframe
        print("   Inner loop running...")
        best_parameters = best_parameters.append(result.best_params_,ignore_index = True)
    if i == 9:
        print("Done")

# Save the best hyperparameters into a dataset for future use
best_parameters.to_csv("RandomSearch_Best_Parameters_rawdata.csv")

# See the frequency of best hyperparameter in Random Search
for col in best_parameters.columns:
    print(best_parameters[col].value_counts(),"\n")

## Grid Search for final hyperparameter selection
from sklearn.model_selection import GridSearchCV

# Outer cross-validation
cv_outer = KFold(n_splits=5, shuffle=True, random_state=42)

# Enumerate splits
outer_results = list()

for train_ix, test_ix in cv_outer.split(x):
    # Train and test split in the inner cross-validation set
    x_in_train, x_in_test = x.iloc[train_ix, :], x.iloc[test_ix, :]
    y_in_train, y_in_test = y.iloc[train_ix], y.iloc[test_ix]

    # Configure the cross-validation procedure
    cv_inner = KFold(n_splits=4, shuffle=True, random_state=42)
    # Define the model
    # min_samples_leaf, max_depth and bootstrap are fixed from the most frequent one of the best_parameters
    model = RandomForestClassifier(min_samples_leaf=1, max_depth=20, bootstrap=True, random_state=42)

    # Define search space based on the frequency from best_parameters
    # Total 6 combination
    space = dict()
    space['n_estimators'] = [400, 800, 1200]
    space['min_samples_split'] = [2, 5]
    # Define search
    search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
    # Execute search
    result = search.fit(x_in_train, y_in_train)
    # Get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    # Evaluate model on the hold out dataset
    yhat = best_model.predict(x_in_test)
    # Evaluate the model
    acc = accuracy_score(y_in_test, yhat)
    # Store the result
    outer_results.append(acc)
    # Report progress
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))

# Summarize the estimated performance of the model
# From these 5 models, obtain the most frequent hyperparameters
print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))

# Final model with hand-picked hyperparameters from the 5 best models of Grid Search
final_model = RandomForestClassifier(n_estimators = 400, min_samples_split = 5, min_samples_leaf = 1, max_depth = 20, bootstrap = True, random_state = 42)


# Last 5-fold cross-validation to have overall validation performance
accuracies = list()
accuracies_train = list()
precisions = list()
recalls = list()

# 50 iteration of 5-Fold cross-validation
for i in range(50):
    final_cv = KFold(n_splits = 5, shuffle = True, random_state = i)
    temp_acc_train = list()
    temp_acc = list()
    for train_ix, test_ix in final_cv.split(x):
        # Train and test split in the inner cross-validation set
        x_in_train, x_in_test = x.iloc[train_ix, :], x.iloc[test_ix, :]
        y_in_train, y_in_test = y.iloc[train_ix], y.iloc[test_ix]

        final_model.fit(x_in_train, y_in_train)
        # Get the best performing model fit on the whole training set
        # Evaluate model on the hold out dataset
        ytrain = final_model.predict(x_in_train)
        yhat = final_model.predict(x_in_test)
        # Evaluate the model
        acc_train = accuracy_score(y_in_train, ytrain)
        acc = accuracy_score(y_in_test, yhat)
        accuracies_train.append(acc_train)
        accuracies.append(acc)
        precisions.append(precision_score(y_in_test, yhat, average = "weighted"))
        recalls.append(recall_score(y_in_test, yhat, average = "weighted"))
    print("In progress...{}/50".format(i+1))

print("Average accuracy of H1_UD: ",np.mean(accuracies))
print("Average precision of H1_UD: ",np.mean(precisions))
print("Average recall of H1_UD: ", np.mean(recalls))


# Final model on filtered data
df_filtered = pd.read_csv("Train_117.csv")
df_filtered = df_filtered.T
x_filtered = df_filtered.iloc[5:,:]


# 5-fold cross-validation to have overall validation performance
accuracies_filtered = list()
accuracies_filtered_train = list()
precisions_filtered = list()
recalls_filtered = list()

# 50 iteration of 5-Fold cross-validation
for i in range(50):
    final_cv = KFold(n_splits = 5, shuffle = True, random_state = i)
    for train_ix, test_ix in final_cv.split(x_filtered):
        # Train and test split in the inner cross-validation set
        x_in_train, x_in_test = x_filtered.iloc[train_ix, :], x_filtered.iloc[test_ix, :]
        y_in_train, y_in_test = y.iloc[train_ix], y.iloc[test_ix]

        final_model.fit(x_in_train, y_in_train)
        # Get the best performing model fit on the whole training set
        # Evaluate model on the hold out dataset
        yhat = final_model.predict(x_in_test)
        ytrain = final_model.predict(x_in_train)
        # Evaluate the model
        acc_train = accuracy_score(y_in_train, ytrain)
        acc = accuracy_score(y_in_test, yhat)
        accuracies_filtered_train.append(acc_train)
        accuracies_filtered.append(acc)
        precisions_filtered.append(precision_score(y_in_test, yhat, average = "weighted"))
        recalls_filtered.append(recall_score(y_in_test, yhat, average = "weighted"))
    print("In progress...{}/50".format(i+1))

print("Average accuracy of H1_FD: ",np.mean(accuracies_filtered))
print("Average precision of H1_FD: ",np.mean(precisions_filtered))
print("Average recall of H1_FD: ", np.mean(recalls_filtered))