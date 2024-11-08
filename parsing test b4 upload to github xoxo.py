import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sb

#pandas initialisation/parameters
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 500)

#parse CSV to DataFrame object
heart_data = pd.read_csv(r"heart.csv")

#rename DataFrame columns with more legible titles
heart_data.columns = ['Age',
                      'Sex',
                      'Chest_pain_type',
                      'Resting_bp', 
                      'Cholesterol',
                      'Fasting_bs',
                      'Resting_ecg', 
                      'Max_heart_rate',
                      'Exercise_induced_angina', 
                      'ST_depression',
                      'ST_slope',
                      'Num_major_vessels',
                      'Thallium_test',
                      'Condition']

#print description of mean/min/max/standard dev
print(heart_data.describe())
