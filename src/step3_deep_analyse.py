import pandas as pd
from matplotlib.patches import bbox_artist
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot  as plt
import seaborn as sns
import numpy as np

#download csv
path = '../data/raw/Top_12_German_Companies_Financial_Data.csv'
df = pd.read_csv(path)

# Top leaders in financial score
# Take a financial columns
df_unique = df.groupby('Company')[['Revenue','Net Income','Equity','Assets','Liabilities','ROA (%)','ROE (%)','Debt to Equity']].mean().reset_index()
fin_cols = ['Revenue','Net Income','Equity','Assets','Liabilities','ROA (%)','ROE (%)','Debt to Equity']
#Make a Normalization
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_unique[fin_cols]), columns=fin_cols)

# Make a score for Companies : I take values like Revenue,Net Income,Equity,Assets,ROA,ROE as positive points
# and Liabilities and Debt to Equity as Negative

df_scaled['Score'] = (df_scaled['Revenue']+df_scaled['Net Income']+df_scaled['Equity']+df_scaled['Assets']+
                      df_scaled['ROE (%)']+df_scaled['ROA (%)'] - df_scaled['Liabilities']-df_scaled['Debt to Equity'])
#add score to original df
df_unique['Score'] = df_scaled['Score']

#and show top by sorting with score value
top_leaders = df_unique.sort_values(by='Score',ascending=False)
print('Leaders by score:')
print(top_leaders[['Score','Company','Revenue','Net Income','Equity','Assets','Liabilities','ROA (%)','ROE (%)','Debt to Equity']].head(12))
#and Visualization for leaders
plt.figure(figsize = (10,7))
sns.barplot(data=df_unique, x = 'Company', y = 'Score', hue = 'Company', palette = 'viridis')
plt.xticks(rotation = 90)
plt.title('Leaders by score')
plt.xlabel('Company')
plt.ylabel('Score')
plt.show()

#Try to show prediction for Revenue in next year for Companies (Linear Regression)
df['Year'] = pd.to_datetime(df['Period']).dt.year
#create list for prediction
prediction = []
for company in df['Company'].unique():
    company_df = df[df['Company']==company].sort_values('Year')
    X = company_df['Year'].values.reshape(-1,1)
    y = company_df['Revenue'].values
    #model creating
    model = LinearRegression()
    model.fit(X,y)
    next_year = np.array([[X.max() + 1]])
    pred = model.predict(next_year)[0]
    prediction.append({'Company': company,'Predicted_Revenue_next_year':pred})

pred_df = pd.DataFrame(prediction).sort_values(by='Predicted_Revenue_next_year', ascending=False)
print(pred_df.head(12))

#Show a prediction for each company with line plot

plt.figure(figsize=(15,6))
companies = df['Company'].unique()
colors = sns.color_palette('tab10', n_colors=len(companies))

for company in companies:
    numeric_cols = ['Revenue', 'Net Income', 'Liabilities', 'Assets', 'Equity', 'ROA (%)', 'ROE (%)', 'Debt to Equity']
    company_df = df[df['Company'] == company].groupby('Year')[numeric_cols].mean().reset_index().sort_values('Year')
    plt.plot(company_df['Year'], company_df['Revenue'], marker='o', label=company)
    next_year = company_df['Year'].max() + 1
    pred = pred_df.loc[pred_df['Company'] == company, 'Predicted_Revenue_next_year'].values[0]
    plt.scatter(next_year, pred, marker='X', s=100)

plt.xlabel('Year')
plt.ylabel('Revenue')
plt.title('Revenue and predict for next year')

plt.grid(True, axis='y', linestyle='--', alpha=0.6, color='gray')
plt.tight_layout()
plt.legend(bbox_to_anchor=(0.1, 0.51))
plt.show()
