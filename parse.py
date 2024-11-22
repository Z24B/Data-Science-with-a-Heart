import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px
import plotly.figure_factory as pff


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
heart_data.Max_heart_rate = heart_data.Max_heart_rate.astype(float)
heart_data.sort_values('Max_heart_rate')

#I AM IN SO MUCH JOY

def multi_distribution(data, x, y, title="multiple distrubution table",):

    sns.distplot(data[x])
    sns.distplot(data[y])

    plt.title(title, fontsize=15)
    plt.show()

multi_distribution(heart_data, 'Resting_bp', 'Max_heart_rate', 'Resting Blood Pressure (mmHg) Distribution')
multi_distribution(heart_data, 'ST_depression', 'Thallium_test', 'idek what this proves ibr')
