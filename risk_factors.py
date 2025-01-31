import matplotlib.pyplot as plt
import seaborn as sns

'''
This file contains the code for exploring and analysing some of the risk 
factors to consider that are more/less likely to be related to heart disease.
'''

def risk_factors(df):

    #Age vs target
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Target', y='Age', data=df)
    plt.title('Age Distribution by Heart Disease Presence')
    plt.xlabel('Target (0 = No Disease, 1 = Disease Present)')
    plt.ylabel('Age')
    plt.show()

    #MaxHR vs target
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Target', y='MaxHR', data=df)
    plt.title('Maximum Heart Rate by Heart Disease Presence')
    plt.xlabel('Target (0 = No Disease, 1 = Disease Present)')
    plt.ylabel('MaxHR')
    plt.show()
