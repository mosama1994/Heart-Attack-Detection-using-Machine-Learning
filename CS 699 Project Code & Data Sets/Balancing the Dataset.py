from collections import Counter
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import seaborn as sns

df_1 = pd.read_csv("Full Clean Data.csv")

plt.figure(figsize=(20, 15))
cor_1 = df_1.corr()
hmap = sns.heatmap(cor_1, annot=True, cmap=plt.cm.Reds, fmt=".2f")
# hmap.set_xticklabels(hmap.get_xticklabels(), rotation=45)
plt.savefig("Correlation Heat Map.png")
plt.show()

# Scaling the data between 1 and 2
scaler = MinMaxScaler(feature_range=(1, 2))
df = scaler.fit_transform(df_1)

X = df[:, 0:df.shape[1] - 1]
y = df[:, df.shape[1] - 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=600)
print(Counter(y_train))
print(y_train.shape)
print(y_test.shape)

# Testing Data Set (as it is) Unbalanced
df_4 = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
df_4.columns = df_1.columns
df_4.to_csv("Classifiers/Unbalanced Testing Data Set.csv", index=False)

# Over Sampling SMOTE
print("SMOTE")
over_sampler_2 = SMOTE(random_state=699)
X_res, y_res = over_sampler_2.fit_resample(X_train, y_train)
df_2 = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)
df_2.columns = df_1.columns
df_2.to_csv("SMOTE Balanced Training Data Set.csv", index=False)

# Over Sampling BorderlineSMOTE
print("Borderline SMOTE")
sm = BorderlineSMOTE(random_state=699)
X_res_2, y_res_2 = sm.fit_resample(X_train, y_train)
df_3 = pd.concat([pd.DataFrame(X_res_2), pd.DataFrame(y_res_2)], axis=1)
df_3.columns = df_1.columns
df_3.to_csv("Borderline SMOTE Balanced Training Data Set.csv", index=False)
