import pandas as pd
import pandas
#Zivha-for-you

heart_data = pd.read_csv("heart.csv")

#print(heart_data)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
print(heart_data.describe())
=======
import matplotlib.pyplot as plt
import numpy as np

heart_data = pd.read_csv("heart.csv")

#Original Data
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
#print(heart_data.describe())

#Sorted Data
sorted_data = heart_data.sort_index(ascending=True)
sorted_data = sorted_data[['age', 'chol']]
print(sorted_data.describe())

'''heart_data = heart_data.groupby("age")
type(heart_data)

# Output
pandas.core.groupby.generic.DataFrameGroupBy'''

sorted_data.plot(kind='scatter', x='age', y='chol', grid=True)
#plt.axhline(y=np.nanmean(sorted_data.y))
plt.show()
main

