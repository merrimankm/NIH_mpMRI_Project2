# Data Processing
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from sklearn import tree
import csv
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz


df = pd.read_csv(r"T:\MIP\Katie_Merriman\Project2Data\sortedData\SortedPatientData_thresh4_trainVal.csv")

y = df["GroundTruth_EPE"]

X = df.drop(["patient", "area_lesion", "dist_lesion", "flipArea_lesion", "flipDist_lesion", "GroundTruth_EPE", "Set"], axis=1)

model = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=44)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv)
print('All - unweighted: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

model = RandomForestClassifier(n_estimators=50, max_features="auto", class_weight={0: 2, 1: 10, 2: 8, 3: 5}, random_state=44)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv)
print('All - weighted: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# no Flip
X = df[["area_var", "area_outside", "area_inside", "area_dist3D", "area_distXY",
                 "dist_var", "dist_outside", "dist_inside", "dist_dist3D", "dist_distXY"]]

model = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=44)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv)
print('No Flip - unweighted: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

model = RandomForestClassifier(n_estimators=50, max_features="auto", class_weight={0: 2, 1: 10, 2: 8, 3: 5}, random_state=44)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv)
print('No Flip - weighted: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# dist
X = df[["dist_var", "dist_outside", "dist_inside", "dist_dist3D", "dist_distXY", "asymmetry",
                "flipDist_outsideChange", "flipDist_dist3DChange", "flipDist_distXYChange",
                "flipDist_var", "flipDist_outside", "flipDist_inside", "flipDist_dist3D", "flipDist_distXY"]]

model = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=44)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv)
print('Dist - unweighted: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

model = RandomForestClassifier(n_estimators=50, max_features="auto", class_weight={0: 2, 1: 10, 2: 8, 3: 5}, random_state=44)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv)
print('Dist - weighted: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# dist no Flip
X = df[["dist_var", "dist_outside", "dist_inside", "dist_dist3D", "dist_distXY"]]

model = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=44)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv)
print('Dist No Flip - unweighted: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

model = RandomForestClassifier(n_estimators=50, max_features="auto", class_weight={0: 2, 1: 10, 2: 8, 3: 5}, random_state=44)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv)
print('Dist No Flip - weighted: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# area
X = df[["area_var", "area_outside", "area_inside", "area_dist3D", "area_distXY", "asymmetry",
                 "flipArea_outsideChange", "flipArea_dist3DChange", "flipArea_distXYChange",
                 "flipArea_var", "flipArea_outside", "flipArea_inside", "flipArea_dist3D", "flipArea_distXY"]]

model = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=44)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv)
print('Area - unweighted: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

model = RandomForestClassifier(n_estimators=50, max_features="auto", class_weight={0: 2, 1: 10, 2: 8, 3: 5}, random_state=44)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv)
print('Area - weighted: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# area no Flip
X = df[["area_var", "area_outside", "area_inside", "area_dist3D", "area_distXY"]]

model = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=44)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv)
print('Area No Flip - unweighted: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

model = RandomForestClassifier(n_estimators=50, max_features="auto", class_weight={0: 2, 1: 10, 2: 8, 3: 5}, random_state=44)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv)
print('Area No Flip - weighted: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
