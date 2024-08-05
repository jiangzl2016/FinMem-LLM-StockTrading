# Generate summary of news articles and 10-K/10-Q reports
# News summary will be stored to experiment/example_output
from model_wrapper import Model_Factory
import os
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import threading
from dotenv import dotenv_values

config = dotenv_values(".env")
model = \
Model_Factory.create_model('chatgpt',
                           key=config['OPENAI_API_KEY'],
                           model_name="gpt-3.5-turbo")
summary = model.summarize
summary_10k_10q = model.summarize_10k_10q
print(model)

# SOURCE_PATH = "/Users/zhonglingjiang/FinMem-LLM-StockTrading/data-pipeline/Fake-Sample-Data/example_input/Fake-News-Data-for-Each-Stock"
SOURCE_PATH = "/Users/zhonglingjiang/FinMem-LLM-StockTrading/data-pipeline/experiment/input"
DEST_PATH = "/Users/zhonglingjiang/FinMem-LLM-StockTrading/data-pipeline/experiment/output"
TEMP_PATH = "/Users/zhonglingjiang/FinMem-LLM-StockTrading/data-pipeline/experiment/temp_output"
# file_ls = ['AMZN_fake.csv', 'MSFT_fake.csv', 'NFLX_fake.csv','TSLA_fake.csv']
file_ls = ['cleaned_TSLA.csv', 'filing_data.parquet'] # 'cleaned_AMZN.csv',
# file_ls = ['filing_data.parquet']

def process_row(row, lock, df, df_name, column, content_type):
    # Perform summary on the 'body' column for news and 'content' column for 10k10q
    if content_type == '10k10q':
        result = summary_10k_10q(row[column])
    else:    
        result = summary(row[column])

    # Acquire lock before updating the dataframe and saving to CSV
    with lock:
        # Update the 'summary' column
        df.at[row.name, 'summary'] = result

        # Save the dataframe to CSV
        df.to_csv(os.path.join(TEMP_PATH, df_name), index=False)
        print("saving")

    # Release the lock after updating and saving
    lock.release()

# Function to parallelize the summary tasks using threads
def parallel_summary(df, df_name, column, content_type):
    lock = threading.Lock()
    df_copy = df.copy()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each row's summary task to the thread pool
        futures = [executor.submit(process_row, row, lock, df_copy, df_name, column, content_type) for _, row in df.iterrows()]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)
    return df_copy

def process_main(file):
    if file.endswith('.csv'):
        content_type = "news"
        print(f"Processing: {file}")
        df = pd.read_csv(os.path.join(SOURCE_PATH, file))
        df["summary"] = None
        ret = parallel_summary(df, file, 'body', content_type)
        ret.to_csv(os.path.join(DEST_PATH, file))
        print(f"New DF dumpped to {os.path.join(DEST_PATH, file)}")
    # Summarize 10-K/10-Q reports, which is in parquet format. Save the result to experiment/example_output
    elif file.endswith('.parquet'):
        content_type = "10k10q"
        print(f"Processing: {file}")
        df = pd.read_parquet(os.path.join(SOURCE_PATH, file))
        df["summary"] = None
        ret = parallel_summary(df, file, 'content', content_type)
        ret.to_parquet(os.path.join(DEST_PATH, file))
        print(f"New DF dumpped to {os.path.join(DEST_PATH, file)}")
    else:
        print(f"Invalid file format: {file}")

for file in file_ls:
    process_main(file)