#code moved to python notebook
import matplotlib.pyplot as plt
import seaborn as sns

'''
This file is used to gather more information from existing data.
Understand the statistics by using visualization techniques such as plotting
pie charts and bar plots on different variables.
'''

def plot_visual(df):

    # Visualize class distribution
    sns.countplot(x='Target', data=df)
    plt.title('Target Distribution')
    plt.xlabel('Target (0 = No Disease, 1 = Disease Present)')
    plt.ylabel('Number of patients')
    plt.show()
    
    #Gender distribution
    gender_counts = df['Sex'].value_counts()
    labels = ['Male', 'Female']
    plt.figure(figsize=(6, 6))
    plt.pie(gender_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['blue', 'pink'])
    plt.title('Gender Distribution')
    plt.show()

    #Max heart rate for patients with/without heart disease
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Target', y='MaxHR', data=df)
    plt.title('Maximum Heart Rate by Heart Disease Presence')
    plt.xlabel('Target (0 = No Disease, 1 = Disease Present)')
    plt.ylabel('Maximum Heart Rate (MaxHR)')
    plt.show()

    #Chest pain type in relation the presence of heart disease
    plt.figure(figsize=(10, 6))
    sns.countplot(x='ChestPainType', hue='Target', data=df, palette='coolwarm')
    plt.title('Chest Pain Type by Heart Disease Presence')
    plt.xlabel('Chest Pain Type')
    plt.ylabel('Number of patients')
    plt.legend(title='Target', labels=['No Disease', 'Disease Present'])
    plt.show()


    #Pairplot of various relationships
    columns = ['Age', 'RestingBP', 'Cholesterol', 'OldPeak', 'Target']
    sns.pairplot(df[columns], hue='Target', diag_kind='kde', palette='husl')
    plt.title('Pairplot of Selected Features by Heart Disease Presence')
    plt.show()

