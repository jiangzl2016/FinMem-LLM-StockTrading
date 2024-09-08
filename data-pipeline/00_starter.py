# Create a price_data.parquet file in the data/03_primary directory. This file will be used to store the stock price data that will be used in the reinforcement learning environment. The file should contain the following columns:
# est_time, equity
# The date range should be between 2021-08-17 and 2023-04-10
# The equity will be TSLA

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy
import datetime
import argparse

# Create a DataFrame
# df = pd.DataFrame({
#     'est_time': pd.date_range(start='2021-08-17', end='2023-04-10', freq='D'),
#     'equity': 'TSLA'
# })

# Write the DataFrame to a parquet file
# table = pa.Table.from_pandas(df)
# pq.write_table(table, 'data/03_primary/price_data.parquet')

if __name__ == "__main__":
    # The script will take in 1) the start date, 2) the end date, and 3) the equity as arguments 4) the output file path
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date", help="The start date")
    parser.add_argument("-e", "--end_date", help="The end date")
    parser.add_argument('-t', "--equity", help="The equity")
    parser.add_argument('-o', "--output_file", help="The output file path")

    args = parser.parse_args()

    df = pd.DataFrame({
        'est_time': pd.date_range(start=args.start_date, end=args.end_date, freq='D'), 
        'equity': args.equity
    })

    table = pa.Table.from_pandas(df)
    pq.write_table(table, args.output_file)


