import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score,  balanced_accuracy_score, confusion_matrix,  recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("creditcard.csv")
pd.set_option('display.max_columns', None)
C = len(df[df["Class"] == 1])/len(df)*100
N = len(df[df["Class"] == 0])/len(df)*100
print(f"Percentage of fraud {C}, Percentage of normal {N}")
print(len(df))


r_scaler = RobustScaler()
s_scaler = StandardScaler()
df['scaled_amount'] = s_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = s_scaler.fit_transform(df['Time'].values.reshape(-1,1))


# # To show how distributed the datas are before scalling
# fig, ax = plt.subplots(2, 2, figsize=(18,4))
#
# amount_val_old = df['Amount'].values
# time_val_old = df['Time'].values
# amount_val_new = df['scaled_amount'].values
# time_val_new = df['scaled_time'].values
# print(type(time_val_new))
#
# sns.distplot(amount_val_old, ax=ax[0, 0], color='r')
# ax[0, 0].set_title('Distribution of Transaction Amount(Old)', fontsize=14)
# ax[0, 0].set_xlim([min(amount_val_old), max(amount_val_old)])
#
# sns.distplot(time_val_old, ax=ax[0, 1], color='b')
# ax[0, 1].set_title('Distribution of Transaction Time(Old)', fontsize=14)
# ax[0, 1].set_xlim([min(time_val_old), max(time_val_old)])
#
# sns.distplot(amount_val_new, ax=ax[1, 0], color='r')
# ax[1, 0].set_title('Distribution of Transaction (New)', fontsize=14)
# ax[1, 0].set_xlim([-10, 10])
#
# sns.distplot(time_val_new, ax=ax[1, 1], color='b')
# ax[1, 1].set_title('Distribution of Transaction Time(New)', fontsize=14)
# ax[1, 1].set_xlim([-10, 10])
# plt.show()
#
#
# # Make sure we use the subsample in our correlation
# f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))
#
# # Entire DataFrame
# corr = df.corr()
# sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
# ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)
#
# sub_sample_corr = df.corr()
# sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
# ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
# plt.show()


scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']
df.drop(['Time','Amount', 'scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)
print(df.head(10))


training, test = train_test_split(df, test_size=0.3, random_state=42)

Y_train = training['Class']
X_train = training.drop('Class', axis=1)

Y_test = test['Class']
X_test = test.drop('Class', axis=1)


##Undersampling Data (Since it is imbalance)
def undersampling(df1):
    df1 = df.sample(frac=1)

    fraud_df = df1.loc[df['Class'] == 1]
    non_fraud_df = df1.loc[df['Class'] == 0][:len(fraud_df)]

    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

    undersampled_df = normal_distributed_df.sample(frac=1, random_state=42)

    return undersampled_df


X_train_undersampled = undersampling(X_train)
Y_train_undersampled = undersampling(Y_train)
print(len(X_train_undersampled))
print(len(Y_train_undersampled))

#Using SMOTE Oversampling Technique
sm = SMOTE(random_state=42)
X_train_SMOTE, Y_train_SMOTE = sm.fit_resample(X_train, Y_train)
print('Original dataset shape %s' % Counter(Y_train))
print('Resampled dataset shape %s' % Counter(Y_train_SMOTE))


###Training Data
def training_data(x_t, y_t):
    #Random Forest
    clf_rf = RandomForestClassifier()
    clf_rf.fit(x_t, y_t)
    y_predict_rf = clf_rf.predict(X_test)

    #Decision Tree
    clf_dec = DecisionTreeClassifier()
    clf_dec.fit(x_t, y_t)
    y_predict_dec = clf_dec.predict(X_test)

    #Naive-Bayes
    clf_gnb = GaussianNB()
    clf_gnb.fit(x_t, y_t)
    y_predict_gnb = clf_gnb.predict(X_test)

    #Linear SVM
    clf_svm = SVC(kernel='linear')
    clf_svm.fit(x_t, y_t)
    y_predict_svm = clf_svm.predict(X_test)

    return y_predict_rf, y_predict_dec, y_predict_gnb, y_predict_svm

# Normal
y_pred_rf_nom, y_pred_dec_nom, y_pred_gnb_nom, y_pred_sv_nom = training_data(X_train, Y_train)

#Undersampling Training
y_pred_rf_us, y_pred_dec_us, y_pred_gnb_us, y_pred_sv_us = training_data(X_train_undersampled, Y_train_undersampled)

#SMOTE
y_pred_rf_SMOTE, y_pred_dec_SMOTE, y_pred_gnb_SMOTE, y_pred_sv_SMOTE = training_data(X_train_SMOTE, Y_train_SMOTE)



def evaluation(y_test, y_predict_rf, y_predict_dec, y_predict_gnb, y_predict_svm):
    #Balanced Accuracy; For Imbalance Dataset
    print("Random Forest Accuracy", balanced_accuracy_score(y_test, y_predict_rf))
    print("Decision Tree Accuracy", balanced_accuracy_score(y_test, y_predict_dec))
    print("NB Accuracy", balanced_accuracy_score(y_test, y_predict_gnb))
    print("SVM Accuracy", balanced_accuracy_score(y_test, y_predict_svm))

    #F1 SCORE
    print("Random Forest F1", f1_score(y_test, y_predict_rf, average=None))
    print("Decision Tree F1", f1_score(y_test, y_predict_dec, average=None))
    print("NB F1", f1_score(y_test, y_predict_gnb, average=None))
    print("SVM F1", f1_score(y_test, y_predict_svm, average=None))

    #Confusion Matrix
    print("SVM CM", confusion_matrix(y_test, y_predict_rf))
    print("Decision Tree SM",confusion_matrix(y_test, y_predict_dec))
    print("NB CM",confusion_matrix(y_test, y_predict_gnb))
    print("SVM CM",confusion_matrix(y_test, y_predict_svm))

#To compare between random_undersampling, SMOTE, and normal way
print("Evaluation Result: Normal", evaluation(Y_test, y_pred_rf_nom, y_pred_dec_nom, y_pred_gnb_nom, y_pred_sv_nom))
print("Evaluation Result: Undersampling", evaluation(Y_test, y_pred_rf_nom, y_pred_dec_nom, y_pred_gnb_nom, y_pred_sv_nom))
print("Evaluation Result: SMOTE", evaluation(Y_test, y_pred_rf_nom, y_pred_dec_nom, y_pred_gnb_nom, y_pred_sv_nom))

#Customer grouping using range partitioning (sliding window)
#Model Tuning tbc
#Implementation of sliding window to spit out
#Implementation of ANN??
#Model Saving tbc

