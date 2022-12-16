import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np

df_1 = pd.read_csv("SMOTE Balanced Training Data Set.csv")
df_2 = pd.read_csv("Borderline SMOTE Balanced Training Data Set.csv")
df_5 = pd.read_csv("Features Selected.csv")

X_1 = df_1.iloc[:, 0:df_1.shape[1]-1].to_numpy()
Y_1 = df_1.iloc[:, -1].to_numpy()
X_2 = df_2.iloc[:, 0:df_2.shape[1]-1].to_numpy()
Y_2 = df_2.iloc[:, -1].to_numpy()

estimator_1 = LogisticRegression(max_iter=1000)
feat_sel_1 = RFE(estimator_1, n_features_to_select=10, step=1)
X_new_1 = feat_sel_1.fit_transform(X_1, Y_1)
print("Recursive Feature Selection using Logistic Regression")
print("Features Selected:")
print(df_1.columns[feat_sel_1.get_support(indices=True)])

estimator_2 = LogisticRegression(max_iter=1000)
feat_sel_2 = RFE(estimator_2, n_features_to_select=10, step=1)
X_new_2 = feat_sel_2.fit_transform(X_2, Y_2)
print("Recursive Feature Selection using Logistic Regression")
print("Features Selected:")
print(df_2.columns[feat_sel_2.get_support(indices=True)])

df_3 = pd.concat([pd.DataFrame(X_new_1), pd.DataFrame(Y_1)], axis=1)
df_3.columns = df_1.columns[np.append(feat_sel_1.get_support(indices=True), df_1.shape[1]-1)]
df_3.to_csv("Classifiers/SMOTE RFE Data Set.csv", index=False)

df_4 = pd.concat([pd.DataFrame(X_new_2), pd.DataFrame(Y_2)], axis=1)
df_4.columns = df_2.columns[np.append(feat_sel_2.get_support(indices=True), df_2.shape[1]-1)]
df_4.to_csv("Classifiers/Borderline SMOTE RFE Data Set.csv", index=False)

df_5.loc[len(df_5.index)] = ["SMOTE", "Recursive Feature Elimination", str(df_1.columns[feat_sel_1.get_support(indices=True)])]
df_5.loc[len(df_5.index)] = ["Borderline SMOTE", "Recursive Feature Elimination", str(df_2.columns[feat_sel_2.get_support(indices=True)])]

df_5.to_csv("Features Selected.csv", index=False)
