# Replace 'YOUR_API_KEY' with your actual Polygon.io API key
BASE_URL = 'https://api.polygon.io/v2/reference/news'

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from tqdm import tqdm

load_dotenv()

def fetch_news(start_date, ticker):
    params = {
        'ticker': ticker,
        'published_utc': start_date.strftime('%Y-%m-%d'),
        'limit': 1000,  # Maximum limit per request
        'apiKey': os.environ.get("POLYGON_API_KEY"),
        'order': 'asc',
        'sort': 'published_utc'
    }
    
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Error fetching news for {start_date.date()}: {response.status_code}")

    return data['results']


if __name__ == "__main__":
    data = pl.read_parquet(os.path.join("data", "03_primary", "price_data.parquet"))
    query_data = (
        data.select([pl.col("est_time").dt.date().alias("date"), pl.col("equity")])
        .unique()
        .to_dict()
    )

    all_news = []
    args_list = list(zip(query_data["date"], query_data["equity"]))
    with tqdm(total=len(args_list)) as pbar:
        for i, arg in enumerate(args_list):
            date, ticker = arg
            result = fetch_news(date, ticker)
            all_news.extend(result)
            pbar.update(1)
            if (i + 1) % 3000 == 0:
                time.sleep(90)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_news)
    
    # Select relevant columns, including the content summary
    columns_to_keep = ['title', 'publisher', 'author', 'published_utc', 'article_url', 'description']
    df = df[columns_to_keep]
    
    # Rename columns so they match with Alpaca news: Index(['author', 'content', 'datetime', 'source', 'headline', 'url',
    #   'date', 'equity'],
    df['equity'] = ticker
    df['published_utc'] = pd.to_datetime(df['published_utc'])
    df['publisher'] = df['publisher'].apply(lambda x: x['name'] if x else None)
    df = df.rename(columns={
        'publisher': 'source',
        'title': 'headline',
        'published_utc': 'datetime',
        'article_url': 'url',
        'description': 'content'
    })
    
    # Save to csv
    # df.to_csv(args.output_path, index=False) # os.path.join("data", "03_primary", "polygon_news.parquet")
    
    # save df to parquet
    table = pa.Table.from_pandas(df)
    save_path = os.path.join("data", "03_primary", "polygon_news.parquet")
    pq.write_table(table, save_path)

    print(f"Saved {len(df)} news items to {str(save_path)}")