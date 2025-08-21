import pandas as pd

# make a DataFrame with pandas from our csv file
path = '../data/raw/Top_12_German_Companies_Financial_Data.csv'
df = pd.read_csv(path)

#check list of general data
#1. Size of data
print('Shape: ', df.shape)
#2. Names of columns
print('Columns: ', df.columns.tolist())
#3. Let's see first 10 rows
print(df.head(10))
#4. And data types
print('\nData types: ')
print(df.dtypes)
#5. Final lets prepare for cleaning , let's see a number of missing values
print('\nMissing values per column: ')
print(df.isnull().sum())