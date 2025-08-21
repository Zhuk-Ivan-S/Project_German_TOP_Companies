import pandas as pd

# make a DataFrame with pandas from our csv file
path = '../data/raw/Top_12_German_Companies_Financial_Data.csv'
df = pd.read_csv(path)
print('How much duplicated in DataFrame: ',df.duplicated().sum())
print('\nMissing values in DataFrame:\n ',df.isnull().sum())

#lets see Names of 12 Companies:
print('\nCompanies: ', df['Company'].unique())
# in prepare level is interesting to know about basic statistic values
print(df.describe())

#This dataframe is already cleaned. And all info is good prepared for analyse