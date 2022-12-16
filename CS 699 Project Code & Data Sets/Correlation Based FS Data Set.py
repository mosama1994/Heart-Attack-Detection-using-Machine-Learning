import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df_1 = pd.read_csv("SMOTE Balanced Training Data Set.csv")
df_2 = pd.read_csv("Borderline SMOTE Balanced Training Data Set.csv")
df_5 = pd.read_csv("Features Selected.csv")

X_1 = df_1.iloc[:, 0:df_1.shape[1] - 1].to_numpy()
Y_1 = df_1.iloc[:, -1].to_numpy()
X_2 = df_2.iloc[:, 0:df_2.shape[1] - 1].to_numpy()
Y_2 = df_2.iloc[:, -1].to_numpy()

plt.figure(figsize=(12, 10))
cor_1 = df_1.corr()
sns.heatmap(cor_1, annot=True, cmap=plt.cm.Reds)
plt.show()
cor_target_1 = abs(cor_1["HEART_ATTACK"]).tolist()
a_1 = pd.concat([pd.DataFrame(cor_target_1), pd.DataFrame(range(0, len(cor_target_1)))], axis=1)
a_1.columns = ["Column 1", "Column 2"]
a_1 = a_1.sort_values(by=["Column 1"])
sel_1 = a_1.iloc[-11:-1, 1].tolist()
X_new_1 = X_1[:, sel_1]
print("Feature selection using Correlation with Target Variable")
print("Features Selected:")
print(df_1.columns[sel_1])

plt.figure(figsize=(12, 10))
cor_2 = df_2.corr()
sns.heatmap(cor_2, annot=True, cmap=plt.cm.Reds)
plt.show()
cor_target_2 = abs(cor_2["HEART_ATTACK"]).tolist()
a_2 = pd.concat([pd.DataFrame(cor_target_2), pd.DataFrame(range(0, len(cor_target_2)))], axis=1)
a_2.columns = ["Column 1", "Column 2"]
a_2 = a_2.sort_values(by=["Column 1"])
sel_2 = a_2.iloc[-11:-1, 1].tolist()
X_new_2 = X_2[:, sel_2]
print("Feature selection using Correlation with Target Variable")
print("Features Selected:")
print(df_2.columns[sel_2])

df_5.loc[len(df_5.index)] = ["SMOTE", "Correlation Based Feature Selection", str(df_1.columns[sel_1])]
df_5.loc[len(df_5.index)] = ["Borderline SMOTE", "Correlation Based Feature Selection", str(df_2.columns[sel_2])]

sel_1.append(df_1.shape[1] - 1)
sel_2.append(df_2.shape[1] - 1)

df_3 = pd.concat([pd.DataFrame(X_new_1), pd.DataFrame(Y_1)], axis=1)
df_3.columns = df_1.columns[sel_1]
df_3.to_csv("Classifiers/SMOTE Correlation Selected Data Set.csv", index=False)

df_4 = pd.concat([pd.DataFrame(X_new_2), pd.DataFrame(Y_2)], axis=1)
df_4.columns = df_2.columns[sel_2]
df_4.to_csv("Classifiers/Borderline SMOTE Correlation Selected Data Set.csv", index=False)

df_5.to_csv("Features Selected.csv", index=False)
