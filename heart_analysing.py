import pandas as pd
import pandas

heart_data = pd.read_csv("heart.csv")

#print(heart_data)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
print(heart_data.describe())

