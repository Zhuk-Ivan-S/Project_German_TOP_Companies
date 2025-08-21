import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#download csv
path = '../data/raw/Top_12_German_Companies_Financial_Data.csv'
df = pd.read_csv(path)

# Basic statistic
print('-------Basic Statistic------')
print(df.describe())

# First of all lets see Leaders in Revenue values
plt.figure(figsize=(10,6))
sns.barplot(data = df ,x='Company', y = 'Revenue', hue = 'Company', palette = 'viridis')
plt.title('Revenue comparison by company')
plt.xlabel('Companies')
plt.ylabel('Revenue (EUR)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#Now lets analyse Profit comparison by companies
plt.figure(figsize=(10,6))
sns.barplot(data=df, x='Company', y = 'Net Income', hue = 'Company', palette='viridis')
plt.title('Profit comparison by companies')
plt.xlabel('Companies')
plt.ylabel('Profit')
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()

#Now lets analyse Liabilities comparison by companies
plt.figure(figsize=(10,6))
sns.barplot(data=df, x='Company', y = 'Liabilities', hue = 'Company', palette='viridis')
plt.title('Liabilities comparison by companies')
plt.xlabel('Companies')
plt.ylabel('Liabilities')
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()

#Now lets analyse Debt to Equity comparison by companies
plt.figure(figsize=(10,6))
sns.barplot(data=df, x='Company', y = 'Debt to Equity', hue = 'Company', palette='viridis')
plt.title('Debt to Equity comparison by companies')
plt.xlabel('Companies')
plt.ylabel('Debt to Equity')
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()

#Now lets analyse Equity comparison by companies
plt.figure(figsize=(10,6))
sns.barplot(data=df, x='Company', y = 'Equity', hue = 'Company', palette='viridis')
plt.title('Equity comparison by companies')
plt.xlabel('Companies')
plt.ylabel('Equity')
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()

#lets understand standard correlation between finance values
#And take just numerical values
df_num = df.select_dtypes(include=['int64','float64'])

plt.figure(figsize=(10,6))
sns.heatmap(df_num.corr(),annot=True, cmap='coolwarm')
plt.title('Correlation between finance values')
plt.show()
# The -1 to 1 correlation coefficient measures the strength and direction
# of the linear relationship between two variables. A value of
# +1 indicates a perfect positive correlation (an increase in one variable is accompanied
# by an increase in the other), -1 indicates a perfect negative correlation
# (an increase in one variable is accompanied by a decrease in the other), and values
# close to 0 indicate no linear relationship.