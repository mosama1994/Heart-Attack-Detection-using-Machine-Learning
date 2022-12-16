import os
import pandas as pd

data = {
    'Classifier': [],
    'Feature Selection Method': [],
    'Class': [],
    'Accuracy': [],
    'TPR': [],
    'FPR': [],
    'Precision': [],
    'Recall': [],
    'F Score': [],
    'MCC': [],
    'ROC': [],
    'TN': [],
    'FP': [],
    'FN': [],
    'TP': []
}

df_1 = pd.DataFrame(data)
df_2 = pd.DataFrame(data)

df_1.to_csv("SMOTE Metrics.csv", index=False)
df_2.to_csv("Borderline SMOTE Metrics.csv", index=False)

# Ada Boost
print("\nAda Boost")
print("\nAda Boost Correlation Feature Selection")
os.system('python "Ada Boost Correlation Feature Selection".py')
print("\nAda Boost F Score Feature Selection")
os.system('python "Ada Boost F Score Feature Selection".py')
print("\nAda Boost Forward SFS")
os.system('python "Ada Boost Forward SFS".py')
print("\nAda Boost RFE Feature Selection")
os.system('python "Ada Boost RFE Feature Selection".py')
print("\nAda Boost Select from Model Feature Selection")
os.system('python "Ada Boost Select from Model Feature Selection".py')

# Decision Tree
print("\nDecision Tree")
print("\nDecision Tree Correlation Feature Selection")
os.system('python "Decision Tree Correlation Feature Selection".py')
print("\nDecision Tree F Score Feature Selection")
os.system('python "Decision Tree F Score Feature Selection".py')
print("\nDecision Tree Forward SFS")
os.system('python "Decision Tree Forward SFS".py')
print("\nDecision Tree RFE Feature Selection")
os.system('python "Decision Tree RFE Feature Selection".py')
print("\nDecision Tree Select from Model Feature Selection")
os.system('python "Decision Tree Select from Model Feature Selection".py')

# Logistic Regression
print("\nLogistic Regression")
print("\nLogistic Regression Correlation Feature Selection")
os.system('python "Logistic Regression Correlation Feature Selection".py')
print("\nLogistic Regression F Score Feature Selection")
os.system('python "Logistic Regression F Score Feature Selection".py')
print("\nLogistic Regression Forward SFS")
os.system('python "Logistic Regression Forward SFS".py')
print("\nLogistic Regression RFE Feature Selection")
os.system('python "Logistic Regression RFE Feature Selection".py')
print("\nLogistic Regression Select from Model Feature Selection")
os.system('python "Logistic Regression Select from Model Feature Selection".py')

# Naive Bayes
print("\nNaive Bayes")
print("\nNaive Bayes Correlation Feature Selection")
os.system('python "Naive Bayes Correlation Feature Selection".py')
print("\nNaive Bayes F Score Feature Selection")
os.system('python "Naive Bayes F Score Feature Selection".py')
print("\nNaive Bayes Forward SFS")
os.system('python "Naive Bayes Forward SFS".py')
print("\nNaive Bayes RFE Feature Selection")
os.system('python "Naive Bayes RFE Feature Selection".py')
print("\nNaive Bayes Select from Model Feature Selection")
os.system('python "Naive Bayes Select from Model Feature Selection".py')

# Random Forest
print("\nRandom Forest")
print("\nRandom Forest Correlation Feature Selection")
os.system('python "Random Forest Correlation Feature Selection".py')
print("\nRandom Forest F Score Feature Selection")
os.system('python "Random Forest F Score Feature Selection".py')
print("\nRandom Forest Forward SFS")
os.system('python "Random Forest Forward SFS".py')
print("\nRandom Forest RFE Feature Selection")
os.system('python "Random Forest RFE Feature Selection".py')
print("\nRandom Forest Select from Model Feature Selection")
os.system('python "Random Forest Select from Model Feature Selection".py')
