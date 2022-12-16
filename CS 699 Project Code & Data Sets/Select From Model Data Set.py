import pandas as pd
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df_1 = pd.read_csv("SMOTE Balanced Training Data Set.csv")
df_2 = pd.read_csv("Borderline SMOTE Balanced Training Data Set.csv")
df_5 = pd.read_csv("Features Selected.csv")

X_1 = df_1.iloc[:, 0:df_1.shape[1]-1].to_numpy()
Y_1 = df_1.iloc[:, -1].to_numpy()
X_2 = df_2.iloc[:, 0:df_2.shape[1]-1].to_numpy()
Y_2 = df_2.iloc[:, -1].to_numpy()

clf_1 = DecisionTreeClassifier(criterion="entropy")
clf_1.fit(X_1, Y_1)
feat_sel_1 = SelectFromModel(clf_1, prefit=True, max_features=10, threshold=-np.inf)
X_new_1 = feat_sel_1.transform(X_1)
print("Feature selection using Select from Model with Random Forest")
print("Features Selected:")
print(df_1.columns[feat_sel_1.get_support(indices=True)])

clf_2 = DecisionTreeClassifier(criterion="entropy")
clf_2.fit(X_2, Y_2)
feat_sel_2 = SelectFromModel(clf_2, prefit=True, max_features=10, threshold=-np.inf)
X_new_2 = feat_sel_2.transform(X_2)
print("Feature selection using Select from Model with Random Forest")
print("Features Selected:")
print(df_2.columns[feat_sel_2.get_support(indices=True)])

df_3 = pd.concat([pd.DataFrame(X_new_1), pd.DataFrame(Y_1)], axis=1)
df_3.columns = df_1.columns[np.append(feat_sel_1.get_support(indices=True), df_1.shape[1]-1)]
df_3.to_csv("Classifiers/SMOTE Select from Model Data Set.csv", index=False)

df_4 = pd.concat([pd.DataFrame(X_new_2), pd.DataFrame(Y_2)], axis=1)
df_4.columns = df_2.columns[np.append(feat_sel_2.get_support(indices=True), df_2.shape[1]-1)]
df_4.to_csv("Classifiers/Borderline SMOTE Select from Model Data Set.csv", index=False)

df_5.loc[len(df_5.index)] = ["SMOTE", "Select From Model Feature Selection", str(df_1.columns[feat_sel_1.get_support(indices=True)])]
df_5.loc[len(df_5.index)] = ["Borderline SMOTE", "Select From Model Feature Selection", str(df_2.columns[feat_sel_2.get_support(indices=True)])]

df_5.to_csv("Features Selected.csv", index=False)
