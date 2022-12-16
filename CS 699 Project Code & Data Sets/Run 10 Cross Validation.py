import os
import pandas as pd

data = {
    'Classifier': [],
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

df_1.to_csv("10 CV Metrics.csv", index=False)

print("\nAda Boost")
os.system('python "Ada Boost 10 CV".py')

print("\nDecision Tree")
os.system('python "Decision Tree 10 CV".py')

print("\nLogistic Regression")
os.system('python "Logistic Regression 10 Cross Validation".py')

print("\nNaive Bayes")
os.system('python "Naive Bayes 10 CV".py')

print("\nRandom Forest")
os.system('python "Random Forest 10 CV".py')
