import pandas as pd
df = pd.read_csv('data/comment_data.csv')
print(df['label'].value_counts())
