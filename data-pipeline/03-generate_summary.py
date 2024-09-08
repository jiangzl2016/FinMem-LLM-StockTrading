# Generate summary of news articles and 10-K/10-Q reports
# News summary will be stored to experiment/example_output
from model_wrapper import Model_Factory
import os
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import threading
from dotenv import dotenv_values


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

if __name__ == '__main__':
    import argparse
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("-s", "--source_path", help="The source path of the input data.")
    argument_parser.add_argument("-d", "--dest_path", help="The destination path of the output data.")
    argument_parser.add_argument("-t", "--temp_path", help="The temporary path for intermediate data.")
    argument_parser.add_argument("-k", "--ticker", help="The stock ticker symbol")
    argument_parser.add_argument("-v", "--gpt_model_version", help="The version of the GPT model to use.")

    args = argument_parser.parse_args()
    SOURCE_PATH, DEST_PATH, TEMP_PATH = args.source_path, args.dest_path, args.temp_path
    file_ls = [f'cleaned_{args.ticker}.csv', 'filing_data.parquet'] 
    
    config = dotenv_values(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 
                           ".env.copy"))
    # Use gpt 3.5 turbo to summarize news
    model1 = \
    Model_Factory.create_model('chatgpt',
                            key=config['OPENAI_API_KEY'],
                            model_name=args.gpt_model_version)
    # Use gpt 4o to summarize 10k/ 10q
    model2 = \
    Model_Factory.create_model('chatgpt',
                            key=config['OPENAI_API_KEY'],
                            model_name='gpt-4o')
    summary = model1.summarize
    summary_10k_10q = model2.summarize_10k_10q
    
    for file in file_ls:
        process_main(file)