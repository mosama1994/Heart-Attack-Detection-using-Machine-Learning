from collections import Counter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold


def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1]}


df_1 = pd.read_csv("Full Clean Data.csv")
df_2 = pd.read_csv("10 CV Metrics.csv")

# Scaling the data between 1 and 2
scaler = MinMaxScaler(feature_range=(1, 2))
df = scaler.fit_transform(df_1)

X = df[:, 0:df.shape[1] - 1]
y = df[:, df.shape[1] - 1]

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=100)

# Random Forest
model_1 = RandomForestClassifier(random_state=0, criterion='entropy')

plt.figure(figsize=(8, 8))
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.title("ROC Curve")
con_mat_1 = np.zeros((2, 2))
tpr_1 = fpr_1 = precision_1 = recall_1 = fscore_1 = mcc_1 = roc_1 = 0
tpr_2 = fpr_2 = precision_2 = recall_2 = fscore_2 = mcc_2 = roc_2 = 0
tpr_w = fpr_w = precision_w = recall_w = fscore_w = mcc_w = roc_w = 0
accuracy = 0

for i, (train, test) in enumerate(cv.split(X, y)):
    model_1.fit(X[train], y[train])
    prediction = model_1.predict(X[test])
    con_mat_1 = con_mat_1 + confusion_matrix(y[test], prediction)
    accuracy = accuracy + accuracy_score(y[test], prediction)
    fpr, tpr, threshold = roc_curve(y[test], model_1.predict_proba(X[test])[:, 1], pos_label=2)
    n = Counter(y[test])

    tpr_1 = tpr_1 + con_mat_1[0][0] / (con_mat_1[0][0] + con_mat_1[0][1])
    fpr_1 = fpr_1 + 1 - (con_mat_1[0][0] / (con_mat_1[0][0] + con_mat_1[0][1]))
    precision_1 = precision_1 + con_mat_1[0][0] / (con_mat_1[0][0] + con_mat_1[1][0])
    recall_1 = recall_1 + con_mat_1[0][0] / (con_mat_1[0][0] + con_mat_1[0][1])
    fscore_1 = fscore_1 + f1_score(y[test], prediction, pos_label=1)
    mcc_1 = mcc_1 + matthews_corrcoef(y[test], prediction)
    roc_1 = roc_1 + roc_auc_score(y[test], model_1.predict_proba(X[test])[:, 1])

    tpr_2 = tpr_2 + con_mat_1[1][1] / (con_mat_1[1][0] + con_mat_1[1][1])
    fpr_2 = fpr_2 + 1 - (con_mat_1[1][1] / (con_mat_1[1][0] + con_mat_1[1][1]))
    precision_2 = precision_2 + con_mat_1[1][1] / (con_mat_1[0][1] + con_mat_1[1][1])
    recall_2 = recall_2 + con_mat_1[1][1] / (con_mat_1[1][0] + con_mat_1[1][1])
    fscore_2 = fscore_2 + f1_score(y[test], prediction, pos_label=2)
    mcc_2 = mcc_2 + matthews_corrcoef(y[test], prediction)
    roc_2 = roc_2 + roc_auc_score(y[test], model_1.predict_proba(X[test])[:, 1])

    tpr_3 = con_mat_1[0][0] / (con_mat_1[0][0] + con_mat_1[0][1])
    fpr_3 = 1 - (con_mat_1[0][0] / (con_mat_1[0][0] + con_mat_1[0][1]))
    precision_3 = con_mat_1[0][0] / (con_mat_1[0][0] + con_mat_1[1][0])
    recall_3 = con_mat_1[0][0] / (con_mat_1[0][0] + con_mat_1[0][1])
    fscore_3 = f1_score(y[test], prediction, pos_label=1)
    mcc_3 = matthews_corrcoef(y[test], prediction)
    roc_3 = roc_auc_score(y[test], model_1.predict_proba(X[test])[:, 1])

    tpr_4 = con_mat_1[1][1] / (con_mat_1[1][0] + con_mat_1[1][1])
    fpr_4 = 1 - (con_mat_1[1][1] / (con_mat_1[1][0] + con_mat_1[1][1]))
    precision_4 = con_mat_1[1][1] / (con_mat_1[0][1] + con_mat_1[1][1])
    recall_4 = con_mat_1[1][1] / (con_mat_1[1][0] + con_mat_1[1][1])
    fscore_4 = f1_score(y[test], prediction, pos_label=2)
    mcc_4 = matthews_corrcoef(y[test], prediction)
    roc_4 = roc_auc_score(y[test], model_1.predict_proba(X[test])[:, 1])

    tpr_w = tpr_w + ((n[1] * tpr_3) + (n[2] * tpr_4)) / (n[1] + n[2])
    fpr_w = fpr_w + ((n[1] * fpr_3) + (n[2] * fpr_4)) / (n[1] + n[2])
    precision_w = precision_w + ((n[1] * precision_3) + (n[2] * precision_4)) / (n[1] + n[2])
    recall_w = recall_w + ((n[1] * recall_3) + (n[2] * recall_4)) / (n[1] + n[2])
    fscore_w = fscore_w + ((n[1] * fscore_3) + (n[2] * fscore_4)) / (n[1] + n[2])
    mcc_w = mcc_w + ((n[1] * mcc_3) + (n[2] * mcc_4)) / (n[1] + n[2])
    roc_w = roc_w + ((n[1] * roc_3) + (n[2] * roc_4)) / (n[1] + n[2])

    plt.plot(fpr, tpr)

plt.savefig("Logistic Regression 10 CV ROC Curve.png")
plt.show()
print(con_mat_1)

print("Accuracy: " + str(accuracy / 10))

print("For class HEART_ATTACK = 1 (No Heart Attack)")
print("TP Rate: " + str(tpr_1 / 10))
print("FP Rate: " + str(fpr_1 / 10))
print("Precision: " + str(precision_1 / 10))
print("Recall: " + str(recall_1 / 10))
print("F Score: " + str(fscore_1 / 10))
print("MCC: " + str(mcc_1 / 10))
print("ROC: " + str(roc_1 / 10))
print()

print("For class HEART_ATTACK = 2 (Yes Heart Attack)")
print("TP Rate: " + str(tpr_2 / 10))
print("FP Rate: " + str(fpr_2 / 10))
print("Precision: " + str(precision_2 / 10))
print("Recall: " + str(recall_2 / 10))
print("F Score: " + str(fscore_2 / 10))
print("MCC: " + str(mcc_2 / 10))
print("ROC: " + str(roc_2 / 10))

print("\nWeighted Scores")
print("TP Rate: " + str(tpr_w / 10))
print("FP Rate: " + str(fpr_w / 10))
print("Precision: " + str(precision_w / 10))
print("Recall: " + str(recall_w / 10))
print("F Score: " + str(fscore_w / 10))
print("MCC: " + str(mcc_w / 10))
print("ROC: " + str(roc_w / 10))

df_2.loc[len(df_2.index)] = ['Random Forest', 'Heart Attack = 1', accuracy / 10, tpr_1 / 10, fpr_1 / 10, precision_1 / 10, recall_1 / 10, fscore_1 / 10, mcc_1 / 10, roc_1 / 10, con_mat_1[0][0], con_mat_1[0][1], con_mat_1[1][0], con_mat_1[1][1]]
df_2.loc[len(df_2.index)] = ['Random Forest', 'Heart Attack = 2', accuracy / 10, tpr_2 / 10, fpr_2 / 10, precision_2 / 10, recall_2 / 10, fscore_2 / 10, mcc_2 / 10, roc_2 / 10, con_mat_1[0][0], con_mat_1[0][1], con_mat_1[1][0], con_mat_1[1][1]]
df_2.loc[len(df_2.index)] = ['Random Forest', 'Weighted', accuracy / 10, tpr_w / 10, fpr_w / 10, precision_w / 10, recall_w / 10, fscore_w / 10, mcc_w / 10, roc_w / 10, con_mat_1[0][0], con_mat_1[0][1], con_mat_1[1][0], con_mat_1[1][1]]

df_2.to_csv("10 CV Metrics.csv", index=False)
