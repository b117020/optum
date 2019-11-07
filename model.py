# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:44:16 2019

@author: Devdarshan
"""

import csv
import os

from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
from sklearn import ensemble
from sklearn import linear_model
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

COLOR = "#E69F00"
COLOR2 = "#56B4E9"

TEST_RATIO = 0.1

STATIC_FIELDS = ['Age', 'Gender', 'Height', 'ICUType']
FIELDS = ['Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol',
          'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT',
          'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg', 'Na', 'NIDiasABP',
          'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
          'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC',
          'Weight']  # Weight is both a static and a time-series field, so may be -1.
NAN_REPLACE = -100


def set_features_to_nan(fieldname):
  """
  Feature values to add to dataset if variable `fieldname` was not observed.
  """
  field_features = {}
  field_features['{}_min'.format(fieldname)] = NAN_REPLACE
  field_features['{}_max'.format(fieldname)] = NAN_REPLACE
  field_features['{}_mean'.format(fieldname)] = NAN_REPLACE
  field_features['{}_first'.format(fieldname)] = NAN_REPLACE
  field_features['{}_last'.format(fieldname)] = NAN_REPLACE
  field_features['{}_diff'.format(fieldname)] = NAN_REPLACE
  return field_features


def featurize(data):
  """ Create features from time-series data. """
  features = {}
  missing_weight = False
  for fieldname in STATIC_FIELDS:
    # Static fields use -1 to denote that the value was not measured.
    if data[fieldname][0][1] == -1:
      features[fieldname] = NAN_REPLACE
    else:
      features[fieldname] = float(data[fieldname][0][1])
  for fieldname in FIELDS:
    # Time-series fields may or may not be measured, but if they are present
    # in the dataset, then the value will be valid (i.e. nonnegative).
    if fieldname in data:
      values = [float(d[1]) for d in data[fieldname]]
      if -1 in values and fieldname == 'Weight':
        # Record that weight was missing for this record id.
        missing_weight = True
        field_features = set_features_to_nan(fieldname)
      else:
        field_features = {}
        field_features['{}_min'.format(fieldname)] = min(values)
        field_features['{}_max'.format(fieldname)] = max(values)
        field_features['{}_mean'.format(fieldname)] = np.mean(values)
        field_features['{}_first'.format(fieldname)] = values[0]
        field_features['{}_last'.format(fieldname)] = values[-1]
        field_features['{}_diff'.format(fieldname)] = values[-1] - values[0]
    else:
      field_features = set_features_to_nan(fieldname)
    features.update(field_features)
  return features, missing_weight


def read_data(outcomes):
  input_data = {}
  for record_id in outcomes.keys():
    filename = '{}.txt'.format(record_id)
    f = open(os.path.join('set-a', filename))
    reader = csv.reader(f)
    # skip header
    next(reader)
    # skip record id
    next(reader)
    data = defaultdict(list)
    for row in reader:
      data[row[1]].append((row[0], row[2]))
    input_data[record_id] = data
    f.close()
  print ('# of records:', len(input_data))
  return input_data


def plot_auc(fpr, tpr):
  plt.plot(fpr, tpr, 's-', lw=1)
  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.show()


def plot_precision_recall(precision, recall):
  plt.plot(precision, recall, 's-', lw=1)
  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.xlabel('Precision')
  plt.ylabel('Recall')
  plt.show()


def train_and_evaluate_logistic_regression_model(X_train, X_test,
                                                 y_train, y_test,
                                                 show_plots=True):

  # Feature preprocessing
  # Impute missing values using mean
  imp = Imputer(missing_values=NAN_REPLACE, strategy='mean', axis=0)
  X_train = imp.fit_transform(X_train)
  X_test = imp.transform(X_test)

  # Scale features
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # Train model
  print ('training model...')
 
  model = linear_model.LogisticRegression()
  model.fit(X_train, y_train)


  # Evaluate model
  print ('Model score (accuracy):', model.score(X_test, y_test))
  predicted = model.predict(X_test)
  predicted_probs = model.predict_proba(X_test)
  fpr, tpr, thresholds = roc_curve(y_test, predicted_probs[:,1])
  if show_plots:
    plot_auc(fpr,tpr)

  print ('AUC', roc_auc_score(y_test, predicted_probs[:,1]))
  precision, recall, thresholds = (
    precision_recall_curve(y_test, predicted_probs[:,1]))
  
  if show_plots:
    plot_precision_recall(precision, recall)


def train_and_evaluate_tree_model(X_train, X_test, y_train, y_test,
                                  show_plots=True):

  # Train model
  print ('training model...')
  model = ensemble.GradientBoostingClassifier(n_estimators=100,
                                              max_depth=15,
                                              max_features='sqrt')
  model.fit(X_train, y_train)
  
  # Evaluate model
  print ('Model score (accuracy):', model.score(X_test, y_test))
  predicted = model.predict(X_test)
  predicted_probs = model.predict_proba(X_test)
  fpr, tpr, thresholds = roc_curve(y_test, predicted_probs[:,1])
  if show_plots:
    plot_auc(fpr, tpr)
 
  print ('AUC', roc_auc_score(y_test, predicted_probs[:,1]))
  precision, recall, thresholds = (
    precision_recall_curve(y_test, predicted_probs[:,1]))
 
  if show_plots:
    plot_precision_recall(precision, recall)


if __name__ == '__main__':

  outcomes = {}
  with open('Outcomes-a.txt', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
      outcomes[row[0]] = row

  # Read input features.
  input_data = read_data(outcomes)

  # Create features and labels.
  features = []
  labels = []
  ids = []
  for id, data in input_data.items():
    feats, missing_weight = featurize(data)
    features.append(feats)
    labels.append(int(outcomes[id][5])) # in-hospital_death
    ids.append(id)

  v = DictVectorizer()
  features = v.fit_transform(features).toarray()
  labels = np.array(labels)

  # Print out some stats about dataset.
  num_train = int(len(labels) * (1 - TEST_RATIO))
  print ('# of test records:', len(features) - num_train)
  print ('Proportion of patients who died:', labels.mean())
  # Train model
  skf = StratifiedKFold()
  for train, test in skf.split(labels,labels):
    X_train = features[train]
    y_train = labels[train]
    X_test = features[test]
    y_test = labels[test]
    train_and_evaluate_tree_model(X_train, X_test, y_train, y_test)
    train_and_evaluate_logistic_regression_model(X_train, X_test, y_train, y_test)