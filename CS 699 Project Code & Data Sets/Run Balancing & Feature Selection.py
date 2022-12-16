import os
import pandas as pd

data = {
    'Balancing Method': [],
    'Classifier': [],
    'Features Selected': [],
}

df_1 = pd.DataFrame(data)

df_1.to_csv("Features Selected.csv", index=False)

os.system('python "Balancing the Dataset".py')
os.system('python "Correlation Based FS Data Set".py')
os.system('python "F Score Attribute Selection Data Set".py')
os.system('python "Forward SFS Data Set".py')
os.system('python "RFE FS Data Set".py')
os.system('python "Select From Model Data Set".py')
