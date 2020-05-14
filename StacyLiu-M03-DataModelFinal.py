# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:38:37 2020

@author: sliu
"""
#Objective: predict severity of cancer based on mammographic data from UCI

# import required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVR
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from warnings import simplefilter
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib

# Download the data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
Mamm = pd.read_csv(url, header=None)
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"] 

# Check the first rows of the data frame
Mamm.head()

#shape of the data frame
Mamm.shape
#(961,6)

# Check the data types
Mamm.dtypes

# Plot the counts for Age
Mamm.loc[:,"Age"].value_counts().plot(kind='bar')
plt.hist(Mamm.loc[:,"Age"], bins = 20, color=[0, 0, 0, 1])
#Age is a numeric variable skewed right

#create histogram of BI-RADS
plt.hist(Mamm.loc[:, "BI-RADS"])
plt.show()
#BI-RADS is a numeric variable centered around 4-5 with outlier at 55

#create histogram of Shape
plt.hist(Mamm.loc[:, "Shape"])
plt.show()
#shape is a categorical variable with missing value

#create histogram of Margin
plt.hist(Mamm.loc[:, "Margin"])
plt.show()
#margin is a shapre variable with missing value

#create histogram of Density
plt.hist(Mamm.loc[:, "Density"])
plt.show()
#shape is a numeric variable with missing value

#create histogram of Severity
plt.hist(Mamm.loc[:, "Severity"])
plt.show()
#shape is a categorical variable we are trying to predict


#function to corece to numeric and imput medians for missing values
def fill_median(x):
   # a=pd.DataFrame(data=x)
    #convert to numeric data including nans
    a= pd.to_numeric(x, errors='coerce')
    #determine the elements that have nan values
    HasNan=np.isnan(a)
    #determine the median
    np.median(a[~HasNan])
    #replacement value for Nans 
    Median=np.nanmedian(a)
    #replace nan with median
    a[HasNan]=Median
    return a
    
#function to replace outliers with mean
def replace_outlier(x):
    a=np.array(x)
    #The high limit for acceptable values is the mean plus 2 standard deviations 
    LH=np.mean(a)+ 2*np.std(a)
    #The low limit for acceptable values is the mean minus 2 standard deviations 
    LL=np.mean(a)- 2*np.std(a)
    # Create Flag for values outside of limits
    FlagBad = (a < LL) | (a > LH)
    # FlagGood is the complement of FlagBad
    FlagGood = ~FlagBad
    # Replace outleiers with the mean of non-outliers
    a[FlagBad] = np.mean(a[FlagGood])
    return a

#call on fill median and replace outliers functions for numeric values
Mamm.loc[:,"BI-RADS"]=fill_median(Mamm.loc[:,"BI-RADS"])
Mamm.loc[:,"BI-RADS"]=replace_outlier(Mamm.loc[:, "BI-RADS"])
Mamm.loc[:,"Age"]=fill_median(Mamm.loc[:,"Age"])
Mamm.loc[:,"Age"]=replace_outlier(Mamm.loc[:, "Age"])
Mamm.loc[:,"Density"]=fill_median(Mamm.loc[:,"Density"])
Mamm.loc[:,"Density"]=replace_outlier(Mamm.loc[:, "Density"])

# Check the data types
Mamm.dtypes
 
# Z-Normalize age
offset = np.mean(Mamm.loc[:,"Age"])
spread = np.std(Mamm.loc[:,"Age"])
xNorm = (Mamm.loc[:,"Age"] - offset)/spread

plt.hist(Mamm.loc[:,"Age"], bins = 20, color=[0, 0, 0, 1])
plt.title("Original Distribution of Age")
plt.show()

plt.hist(xNorm, bins = 20, color=[1, 1, 0, 1])
plt.title("Z-normalization of Age")
plt.show()

#replace Age with the znormalized Age
Mamm.loc[:,"Age"]=xNorm

# Check the first rows of the data frame
Mamm.head()

# Decode category columns 
Mamm.loc[ Mamm.loc[:, "Shape"] == "1", "Shape"] = "round"
Mamm.loc[Mamm.loc[:, "Shape"] == "2", "Shape"] = "oval"
Mamm.loc[Mamm.loc[:, "Shape"] == "3", "Shape"] = "lobular"
Mamm.loc[Mamm.loc[:, "Shape"] == "4", "Shape"] = "irregular"
Mamm.loc[Mamm.loc[:, "Shape"] == "?", "Shape"] = "irregular"
Mamm.loc[Mamm.loc[:, "Margin"] == "1", "Margin"] = "circumscribed"
Mamm.loc[Mamm.loc[:, "Margin"] == "2", "Margin"] = "microlobulated"
Mamm.loc[Mamm.loc[:, "Margin"] == "3", "Margin"] = "obscured"
Mamm.loc[Mamm.loc[:, "Margin"] == "4", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "5", "Margin"] = "spiculated"
Mamm.loc[Mamm.loc[:, "Margin"] == "?", "Margin"] = "circumscribed"
####################

# Check the first rows of the data frame
Mamm.head()
####################

# Get the counts for each value
Mamm.loc[:,"Shape"].value_counts()
Mamm.loc[:,"Margin"].value_counts()
####################

# Plot the counts for each category
Mamm.loc[:,"Shape"].value_counts().plot(kind='bar')
####################

# Simplify Shape by consolidating oval and round
Mamm.loc[Mamm.loc[:, "Shape"] == "round", "Shape"] = "oval"

# Plot the counts for each category
Mamm.loc[:,"Shape"].value_counts().plot(kind='bar')
plt.show()
####################

# Plot the counts for each category
Mamm.loc[:,"Margin"].value_counts().plot(kind='bar')
####################

# Simplify Margin by consolidating ill-defined, microlobulated, and obscured
Mamm.loc[Mamm.loc[:, "Margin"] == "microlobulated", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "obscured", "Margin"] = "ill-defined"

# Plot the counts for each category
Mamm.loc[:,"Margin"].value_counts().plot(kind='bar')
plt.show()

# Create 3 new columns, one for each state in "Shape"
Mamm.loc[:, "oval"] = (Mamm.loc[:, "Shape"] == "oval").astype(int)
Mamm.loc[:, "lobul"] = (Mamm.loc[:, "Shape"] == "lobular").astype(int)
Mamm.loc[:, "irreg"] = (Mamm.loc[:, "Shape"] == "irregular").astype(int)
##############

# Remove obsolete column
Mamm = Mamm.drop("Shape", axis=1)
##############

# Create 3 new columns, one for each state in "Margin"
Mamm.loc[:, "ill-d"] = (Mamm.loc[:, "Margin"] == "ill-defined").astype(int)
Mamm.loc[:, "circu"] = (Mamm.loc[:, "Margin"] == "circumscribed").astype(int)
Mamm.loc[:, "spicu"] = (Mamm.loc[:, "Margin"] == "spiculated").astype(int)
##############

# Remove obsolete column
Mamm = Mamm.drop("Margin", axis=1)

# Check the first rows of the data frame
Mamm.head()

# Check the data types
Mamm.dtypes

#####################
#Unsupervised Learning: Kmeans

# Create Points to cluster
Points = pd.DataFrame()
Points.loc[:,0] = Mamm.loc[:,"BI-RADS"]
Points.loc[:,1] = Mamm.loc[:,"Age"]

# Compare distributions of the two dimensions
plt.hist(Points.loc[:,0], bins = 20, color=[0, 0, 1, 0.5])
plt.hist(Points.loc[:,1], bins = 20, color=[1, 1, 0, 0.5])
plt.title("Compare Distributions")
plt.show()

#apply kmeans
kmeans= KMeans(n_clusters=4)
y_kmeans = kmeans.fit_predict(Points)
print(y_kmeans)
print(kmeans.cluster_centers_)


#add kmeans cluster label to the dataset
Mamm.loc[:, "cluster"] = pd.Series(y_kmeans)

# Check the first rows of the data frame
Mamm.head()

# Check the data types
Mamm.dtypes

################
#supervised Learning

#Question: Is the tumor severe?

#set severity as the target variable
Mamm_severity=Mamm.pop("Severity")

# split data set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Mamm, Mamm_severity, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#logistic regression to predict severity
print ('\n Use logistic regression to predict species from other variables')
clf = LogisticRegression()
clf.fit(X_train, y_train)
BothProbabilities = clf.predict_proba(X_test)
probabilities = BothProbabilities[:,1]

probabilities

print ('\nConfusion Matrix and Metrics: 0.9 was chosen as the probability threshold to ensure higher degree of precision')
Threshold = 0.9
print ("Probability Threshold is chosen to be:", Threshold)
predictions = (probabilities > Threshold).astype(int)
predictions
CM = confusion_matrix(y_test, predictions)
CM
tn, fp, fn, tp = CM.ravel()
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(y_test, predictions)
print ("Accuracy rate:", np.round(AR, 2))
P = precision_score(y_test, predictions)
print ("Precision:", np.round(P, 2))
R = recall_score(y_test, predictions)
print ("Recall:", np.round(R, 2))
F1 = f1_score(y_test, predictions)
print ("\nF1 score:", np.round(F1, 2))

 # False Positive Rate, True Posisive Rate, probability thresholds
fpr, tpr, th = roc_curve(y_test, probabilities)
AUC = auc(fpr, tpr)

plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('ROC Curve - Logistic Regression')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()

#######################
# Random Forest classifier to predict severity
estimators = 10 # number of trees parameter
mss = 2 # mininum samples split parameter
print ('\n\nRandom Forest classifier\n')
clf = RandomForestClassifier(n_estimators=estimators, min_samples_split=mss) # default parameters are fine
#train the model
clf.fit(X_train, y_train)
print ("predictions for test set:")
#apply trained classifier to the test data
y_pred=clf.predict(X_test)
print (y_pred)
print ('actual class values:')
print (y_test)

CM2 = confusion_matrix(y_test, y_pred)
CM2
tn, fp, fn, tp = CM2.ravel()
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(y_test, y_pred)
print ("Accuracy rate:", np.round(AR, 2))
P = precision_score(y_test, y_pred)
print ("Precision:", np.round(P, 2))
R = recall_score(y_test, y_pred)
print ("Recall:", np.round(R, 2))
F1 = f1_score(y_test, y_pred)
print ("\nF1 score:", np.round(F1, 2))

 # False Positive Rate, True Posisive Rate, probability thresholds
fpr, tpr, th = roc_curve(y_test, y_pred)
AUC = auc(fpr, tpr)

plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('ROC Curve - Random Forest')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()

# Conclusion: To predict the severity of cancer from the Mamm dataset, 
# Logitics regression and random forest models were applied to the cleansed dataset (outliers replaced, missing values decoded
# categorical data consolidated and one hot encoded)
# Logistic Regression produced a higher AUC of 0.94 compared to 0.81 from the random forest model
# therefore logistic regression should be used to predict the severity of cancer.


