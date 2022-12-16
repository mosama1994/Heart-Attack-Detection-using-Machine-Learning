from collections import Counter
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

df_1 = pd.read_csv("SMOTE RFE Data Set.csv")
df_2 = pd.read_csv("Borderline SMOTE RFE Data Set.csv")
df_3 = pd.read_csv("Unbalanced Testing Data Set.csv")
df_4 = pd.read_csv("SMOTE Metrics.csv")
df_5 = pd.read_csv("Borderline SMOTE Metrics.csv")

X_train_1 = df_1.iloc[:, 0:df_1.shape[1]-1].to_numpy()
Y_train_1 = df_1.iloc[:, -1].to_numpy()
X_train_2 = df_2.iloc[:, 0:df_1.shape[1]-1].to_numpy()
Y_train_2 = df_2.iloc[:, -1].to_numpy()
X_test = df_3.iloc[:, 0:df_1.shape[1]-1].to_numpy()
Y_test = df_3.iloc[:, -1].to_numpy()

n = Counter(Y_test)

# Applying Ada Boost Classifier for SMOTE
base = LogisticRegression(max_iter=1000)
clf_1 = AdaBoostClassifier(base_estimator=base, random_state=0)
clf_1.fit(X_train_1, Y_train_1)
prediction_1 = clf_1.predict(X_test)
accuracy_1 = accuracy_score(Y_test, prediction_1)
con_mat_1 = confusion_matrix(Y_test, prediction_1)
print("Accuracy: " + str(accuracy_1 * 100))
print("Confusion Matrix:")
print(con_mat_1)

tpr_1 = con_mat_1[0][0] / (con_mat_1[0][0] + con_mat_1[0][1])
fpr_1 = 1 - (con_mat_1[0][0] / (con_mat_1[0][0] + con_mat_1[0][1]))
precision_1 = con_mat_1[0][0] / (con_mat_1[0][0] + con_mat_1[1][0])
recall_1 = con_mat_1[0][0] / (con_mat_1[0][0] + con_mat_1[0][1])
fscore_1 = f1_score(Y_test, prediction_1, pos_label=1)
mcc_1 = matthews_corrcoef(Y_test, prediction_1)
roc_1 = roc_auc_score(Y_test, clf_1.predict_proba(X_test)[:, 1])

print("For class HEART_ATTACK = 1 (No Heart Attack)")
print("TP Rate: " + str(tpr_1))
print("FP Rate: " + str(fpr_1))
print("Precision: " + str(precision_1))
print("Recall: " + str(recall_1))
print("F Score: " + str(fscore_1))
print("MCC: " + str(mcc_1))
print("ROC: " + str(roc_1))
print()

tpr_2 = con_mat_1[1][1] / (con_mat_1[1][0] + con_mat_1[1][1])
fpr_2 = 1 - (con_mat_1[1][1] / (con_mat_1[1][0] + con_mat_1[1][1]))
precision_2 = con_mat_1[1][1] / (con_mat_1[0][1] + con_mat_1[1][1])
recall_2 = con_mat_1[1][1] / (con_mat_1[1][0] + con_mat_1[1][1])
fscore_2 = f1_score(Y_test, prediction_1, pos_label=2)
mcc_2 = matthews_corrcoef(Y_test, prediction_1)
roc_2 = roc_auc_score(Y_test, clf_1.predict_proba(X_test)[:, 1])

print("For class HEART_ATTACK = 2 (Yes Heart Attack)")
print("TP Rate: " + str(tpr_2))
print("FP Rate: " + str(fpr_2))
print("Precision: " + str(precision_2))
print("Recall: " + str(recall_2))
print("F Score: " + str(fscore_2))
print("MCC: " + str(mcc_2))
print("ROC: " + str(roc_2))

tpr_w1 = ((n[1] * tpr_1) + (n[2] * tpr_2)) / (n[1] + n[2])
fpr_w1 = ((n[1] * fpr_1) + (n[2] * fpr_2)) / (n[1] + n[2])
precision_w1 = ((n[1] * precision_1) + (n[2] * precision_2)) / (n[1] + n[2])
recall_w1 = ((n[1] * recall_1) + (n[2] * recall_2)) / (n[1] + n[2])
fscore_w1 = ((n[1] * fscore_1) + (n[2] * fscore_2)) / (n[1] + n[2])
mcc_w1 = ((n[1] * mcc_1) + (n[2] * mcc_2)) / (n[1] + n[2])
roc_w1 = ((n[1] * roc_1) + (n[2] * roc_2)) / (n[1] + n[2])

print("\nWeighted Scores")
print("TP Rate: " + str(tpr_w1))
print("FP Rate: " + str(fpr_w1))
print("Precision: " + str(precision_w1))
print("Recall: " + str(recall_w1))
print("F Score: " + str(fscore_w1))
print("MCC: " + str(mcc_w1))
print("ROC: " + str(roc_w1))

fpr_5, tpr_5, thresholds_5 = roc_curve(Y_test, clf_1.predict_proba(X_test)[:, 1], pos_label=2)
plt.figure(figsize=(8, 8))
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.title("ROC Curve")
plt.plot(fpr_5, tpr_5)
plt.savefig("ROC Curve Ada Boost RFE Feature Selection (SMOTE).png")
plt.show()

# Applying Ada Boost Classifier for Borderline SMOTE
base = LogisticRegression(max_iter=1000)
clf_2 = AdaBoostClassifier(base_estimator=base, random_state=0)
clf_2.fit(X_train_2, Y_train_2)
prediction_2 = clf_2.predict(X_test)
accuracy_2 = accuracy_score(Y_test, prediction_2)
con_mat_2 = confusion_matrix(Y_test, prediction_2)
print("\nAccuracy: " + str(accuracy_2 * 100))
print("Confusion Matrix:")
print(con_mat_2)

tpr_3 = con_mat_2[0][0] / (con_mat_2[0][0] + con_mat_2[0][1])
fpr_3 = 1 - (con_mat_2[0][0] / (con_mat_2[0][0] + con_mat_2[0][1]))
precision_3 = con_mat_2[0][0] / (con_mat_2[0][0] + con_mat_2[1][0])
recall_3 = con_mat_2[0][0] / (con_mat_2[0][0] + con_mat_2[0][1])
fscore_3 = f1_score(Y_test, prediction_2, pos_label=1)
mcc_3 = matthews_corrcoef(Y_test, prediction_2)
roc_3 = roc_auc_score(Y_test, clf_2.predict_proba(X_test)[:, 1])

print("For class HEART_ATTACK = 1 (No Heart Attack)")
print("TP Rate: " + str(tpr_3))
print("FP Rate: " + str(fpr_3))
print("Precision: " + str(precision_3))
print("Recall: " + str(recall_3))
print("F Score: " + str(fscore_3))
print("MCC: " + str(mcc_3))
print("ROC: " + str(roc_3))
print()

tpr_4 = con_mat_2[1][1] / (con_mat_2[1][0] + con_mat_2[1][1])
fpr_4 = 1 - (con_mat_2[1][1] / (con_mat_2[1][0] + con_mat_2[1][1]))
precision_4 = con_mat_2[1][1] / (con_mat_2[0][1] + con_mat_2[1][1])
recall_4 = con_mat_2[1][1] / (con_mat_2[1][0] + con_mat_2[1][1])
fscore_4 = f1_score(Y_test, prediction_2, pos_label=2)
mcc_4 = matthews_corrcoef(Y_test, prediction_2)
roc_4 = roc_auc_score(Y_test, clf_2.predict_proba(X_test)[:, 1])

print("For class HEART_ATTACK = 2 (Yes Heart Attack)")
print("TP Rate: " + str(tpr_4))
print("FP Rate: " + str(fpr_4))
print("Precision: " + str(precision_4))
print("Recall: " + str(recall_4))
print("F Score: " + str(fscore_4))
print("MCC: " + str(mcc_4))
print("ROC: " + str(roc_4))

tpr_w2 = ((n[1] * tpr_3) + (n[2] * tpr_4)) / (n[1] + n[2])
fpr_w2 = ((n[1] * fpr_3) + (n[2] * fpr_4)) / (n[1] + n[2])
precision_w2 = ((n[1] * precision_3) + (n[2] * precision_4)) / (n[1] + n[2])
recall_w2 = ((n[1] * recall_3) + (n[2] * recall_4)) / (n[1] + n[2])
fscore_w2 = ((n[1] * fscore_3) + (n[2] * fscore_4)) / (n[1] + n[2])
mcc_w2 = ((n[1] * mcc_3) + (n[2] * mcc_4)) / (n[1] + n[2])
roc_w2 = ((n[1] * roc_3) + (n[2] * roc_4)) / (n[1] + n[2])

print("\nWeighted Scores")
print("TP Rate: " + str(tpr_w2))
print("FP Rate: " + str(fpr_w2))
print("Precision: " + str(precision_w2))
print("Recall: " + str(recall_w2))
print("F Score: " + str(fscore_w2))
print("MCC: " + str(mcc_w2))
print("ROC: " + str(roc_w2))

fpr_6, tpr_6, thresholds_6 = roc_curve(Y_test, clf_2.predict_proba(X_test)[:, 1], pos_label=2)
plt.figure(figsize=(8, 8))
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.title("ROC Curve")
plt.plot(fpr_6, tpr_6)
plt.savefig("ROC Curve Ada Boost RFE Feature Selection (BSMOTE).png")
plt.show()

# SMOTE Metrics
df_4.loc[len(df_4.index)] = ['Ada Boost', 'RFE Feature Selection', 'Heart Attack = 1', accuracy_1, tpr_1, fpr_1, precision_1, recall_1, fscore_1, mcc_1, roc_1, con_mat_1[0][0], con_mat_1[0][1], con_mat_1[1][0], con_mat_1[1][1]]
df_4.loc[len(df_4.index)] = ['Ada Boost', 'RFE Feature Selection', 'Heart Attack = 2', accuracy_1, tpr_2, fpr_2, precision_2, recall_2, fscore_2, mcc_2, roc_2, con_mat_1[0][0], con_mat_1[0][1], con_mat_1[1][0], con_mat_1[1][1]]
df_4.loc[len(df_4.index)] = ['Ada Boost', 'RFE Feature Selection', 'Weighted', accuracy_1, tpr_w1, fpr_w1, precision_w1, recall_w1, fscore_w1, mcc_w1, roc_w1, con_mat_1[0][0], con_mat_1[0][1], con_mat_1[1][0], con_mat_1[1][1]]

# Borderline SMOTE
df_5.loc[len(df_5.index)] = ['Ada Boost', 'RFE Feature Selection', 'Heart Attack = 1', accuracy_2, tpr_3, fpr_3, precision_3, recall_3, fscore_3, mcc_3, roc_3, con_mat_2[0][0], con_mat_2[0][1], con_mat_2[1][0], con_mat_2[1][1]]
df_5.loc[len(df_5.index)] = ['Ada Boost', 'RFE Feature Selection', 'Heart Attack = 2', accuracy_2, tpr_4, fpr_4, precision_4, recall_4, fscore_4, mcc_4, roc_4, con_mat_2[0][0], con_mat_2[0][1], con_mat_2[1][0], con_mat_2[1][1]]
df_5.loc[len(df_5.index)] = ['Ada Boost', 'RFE Feature Selection', 'Weighted', accuracy_2, tpr_w2, fpr_w2, precision_w2, recall_w2, fscore_w2, mcc_w2, roc_w2, con_mat_2[0][0], con_mat_2[0][1], con_mat_2[1][0], con_mat_2[1][1]]

df_4.to_csv("SMOTE Metrics.csv", index=False)
df_5.to_csv("Borderline SMOTE Metrics.csv", index=False)
