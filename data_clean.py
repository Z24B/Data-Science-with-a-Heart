import pandas as pd
from sklearn.impute import SimpleImputer

'''
Data preparation(loading data, cleaning up data, check/handle null values, and other relevant information.
'''

def load_and_clean(file_path):

    #Load the dataset
    df = pd.read_csv(file_path)
    print(df.head())

    #Data clean up
    df.rename(columns={
        'age' : 'Age',
        'sex' : 'Sex',
        'cp' : 'ChestPainType',
        'trestbps' : 'RestingBP',
        'chol' : 'Cholesterol',
        'fbs' : 'FastingBS',
        'restecg' : 'RestingECG',
        'thalach' : 'MaxHR',
        'exang' : 'ExerciseAngina',
        'oldpeak' : 'OldPeak',
        'slope' : 'SlopePeakExercise',
        'ca' : 'NumMajorVessels',
        'thal' : 'ThalDisorder',
        'target' : 'Condition'
        }, inplace=True)

    #Check columns for null valuess
    print(df.isnull().sum())

    #Basic information about the dataset
    print(df.info())

    #Basic summary of numerical statistics
    print(df.describe())

    return df
