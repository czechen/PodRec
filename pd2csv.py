import pandas as pd
from yt_url_extraction import yt_url_extraction

df = pd.read_csv('podcasts_data.csv',header=0)

data2append = yt_url_extraction('https://www.youtube.com/watch?v=co_MeKSnyAo&t=24s&ab_channel=LexFridman')

if not df['link'].isin([data2append['link']]).any():
    data_df = pd.DataFrame([data2append])
    df = pd.concat([df, data_df], ignore_index=True)
    print(df)
else:
    print("This podcast is already in the dataset")

df.to_csv('podcasts_data.csv')