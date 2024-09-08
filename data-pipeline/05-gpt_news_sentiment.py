import pickle
from tqdm import tqdm
import datetime
from transformers import BertTokenizer, BertForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
import pandas as pd
import numpy as np
import math
import os
from dotenv import dotenv_values, load_dotenv
import json
from openai import OpenAI
import concurrent.futures

configs = dotenv_values(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 
                           ".env.copy"))

API_KEY = configs['OPENAI_API_KEY']
assert API_KEY, "ERROR: OpenAI Key is missing"
client = OpenAI(
    api_key=API_KEY
)

def subset_symbol_dict(input_dir, cur_symbol):
    new_dict = {}
    with open(input_dir, "rb") as f:
        data = pickle.load(f)
    # Now combined_dict contains all the data from the tuple
    new_dict = {}
    ticker_dict_byDate = {}
    for k, v in tqdm(data.items()):
        cur_price = v[0]['price']  # price
        cur_news = v[1]['news']   # news
        cur_filing_q = v[2]['filing_q']  # form q
        cur_filing_k = v[3]['filing_k']  # form k
        # print('Date: ---------', k)
        # print('Available tickers: ---------',cur_news.keys())

        new_price = {}
        new_filing_k = {}
        new_filing_q = {}
        new_news = {}
        if cur_symbol in list(cur_price.keys()):
            new_price[cur_symbol] = cur_price[cur_symbol]
        if cur_symbol in list(cur_filing_k.keys()):
            new_filing_k[cur_symbol] = cur_filing_k[cur_symbol]
        if cur_symbol in list(cur_filing_q.keys()):
            new_filing_q[cur_symbol] = cur_filing_q[cur_symbol]
        if cur_symbol in list(cur_news.keys()):
            new_news[cur_symbol] = cur_news[cur_symbol]
        else:
            continue

        new_dict[k] = {
            "price": new_price,
            "filing_k": new_filing_k,
            "filing_q": new_filing_q,
            "news": new_news
        }
        ticker_dict_byDate[k] = list(new_dict[k]["price"].keys())
        # print("On date: ", k, "ticker list: ---", ticker_dict_byDate[k])

    return new_dict, ticker_dict_byDate

#### GPT 4o-mini
# Function to analyze sentiment
import json
def sentiment_score_gpt(text, model="gpt-3.5-turbo"):
    if (text is None) or pd.isna(text) or text == "":
        return {"scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}, "prediction": "neutral"}
    try:
        messages = [
            {"role": "system", "content": """You are trained to analyze and detect the sentiment of given financial news text."""},
            {"role": "user", "content": f"""Return normalized scores for each sentiment class (positive, negative, neutral) where the scores sum to 1.0, 
        along with a final prediction of the dominant sentiment.

        Format your response as a JSON object with the following structure:
        {{
        "scores": {{
            "positive": <float>,
            "negative": <float>,
            "neutral": <float>
        }},
        "prediction": <string>
        }}

        The "prediction" should be one of "positive", "negative", or "neutral", representing the dominant sentiment.

        Ensure that the sum of all three scores equals 1.0.

        Financial news:
        {text}"""}
        ]
        
        response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0)
        analysis_result = json.loads(response.choices[0].message.content)
    except:
        print('Error in sentiment_score Text: ', text)
        return {"scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}, "prediction": "neutral"}
    
    return analysis_result

def append_sentiment_score(text, model):
    sentiment_score = sentiment_score_gpt(text, model)
    if "scores" not in sentiment_score:
        return f"The overall sentiment for this news is neutral, where the positive score is 0.0, the negative score is 0.0 and the neutral score is 1.0"
    pos_score = sentiment_score["scores"]["positive"]
    neu_score = sentiment_score["scores"]["neutral"]
    neg_score = sentiment_score["scores"]["negative"]
    sentiment = sentiment_score["prediction"]
    sentiment_sentence = f"The overall sentiment for this news is {sentiment}, where "
    pos_sentence = f"the positive score is {pos_score}, "
    neu_sentence = f"the neutral score is {neu_score}, "
    neg_sentence = f"and the negative score is {neg_score}."
    combine_news_sentiment = f"{text} {sentiment_sentence} {pos_sentence} {neu_sentence} {neg_sentence}"
    return combine_news_sentiment

def parallel_sentiment_score(news, model):
    # 1. Use ThreadPoolExecutor to parallelize the sentiment analysis
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(append_sentiment_score, news_i, model) for news_i in news]
        concurrent.futures.wait(futures)
        results = [future.result() for future in futures]

    return results


    
# The news dictionary is a nested dictionary with the following structure:
# {"datetime.date(2021, 1, 1)": ({'price': {'TSLA': <float>}},
#   {'news': {'TSLA': [<str>, <str>, ...]}},
#   {'filing_q': {}},
#   {'filing_k': {}}),
#  "datetime.date(2021, 1, 2)": ...
# }
# The following function does the following for each news article in the dictionary in parallel:
# 1. Analyzes the sentiment of the news article using the GPT model
# 2. Extracts the positive, negative, neutral scores and the dominant sentiment
# 3. Create a new string, i.e. "The overall sentiment for this news is {sentiment}, where the positive score is {pos_score}, the neutral score is {neu_score}, and the negative score is {neg_score}."
# 4. Appends the new string to the news article
# 5. Updates the news article in the dictionary
# you can use the parallel_sentiment_score function to parallelize the sentiment analysis
def assign_gpt_scores_in_parallel(new_dict, cur_symbol, model):
    for i_date in tqdm(new_dict):
        i_date_dict = new_dict[i_date]
        if len(i_date_dict["news"]) != 0:
            i_date_news = i_date_dict["news"][cur_symbol]
            j_new_news = [] 
            j_new_news = parallel_sentiment_score(i_date_news, model)
            i_date_dict["news"][cur_symbol] = j_new_news

# def assign_gpt_scores(new_dict, cur_symbol, model):
#     for i_date in tqdm(new_dict):
#         i_date_dict = new_dict[i_date]
#         if len(i_date_dict["news"]) != 0:
#             i_date_news = i_date_dict["news"][cur_symbol]
#             j_new_news = []
#             for j in range(len(i_date_news)):
#                 j_news = i_date_news[j]
#                 j_news_sentiment = sentiment_score_gpt(j_news, model)
#                 pos_score = j_news_sentiment["scores"]["positive"]
#                 neu_score = j_news_sentiment["scores"]["neutral"]
#                 neg_score = j_news_sentiment["scores"]["negative"]
#                 sentiment = j_news_sentiment["prediction"]
#                 sentiment_sentence = f"The overall sentiment for this news is {sentiment}, where "
#                 pos_sentence = f"the positive score is {pos_score}, "
#                 neu_sentence = f"the neutral score is {neu_score}, "
#                 neg_sentence = f"and the negative score is {neg_score}."
#                 j_combine_news_sentiment = f"{j_news} {sentiment_sentence} {pos_sentence} {neu_sentence} {neg_sentence}"
#                 j_new_news.append(j_combine_news_sentiment)

#             i_date_dict["news"][cur_symbol] = j_new_news
    

# Function to analyze sentiment

def export_sub_symbol(input_dir, cur_symbol_lst, model_type, output_dir):
    print('Ticker list: ------', cur_symbol_lst)
    for cur_symbol_0 in cur_symbol_lst:
        new_dict, ticker_dict_byDate = subset_symbol_dict(input_dir, cur_symbol_0)
        assign_gpt_scores_in_parallel(new_dict, cur_symbol_0, model_type)    
        out_dir = os.path.join(output_dir, 'subset_symbols_'+ cur_symbol_0 + ".pkl")
    
        with open(out_dir, "wb") as f:
            pickle.dump(new_dict, f)
        print('*************---------------************')
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--ticker", help="The stock ticker symbol.")
    parser.add_argument("-p", "--tokenizer_dir", help="The tokenizer directory.")
    parser.add_argument("-i", "--input_dir", help="input directory")
    parser.add_argument("-o", "--output_dir", help="output directory")
    parser.add_argument("-m", "--model", default="gpt-3.5-turbo", help="The GPT model to use.", required=False)
    args = parser.parse_args()
    
    cur_symbol_lst = [args.ticker]
    tokenizer_dir = args.tokenizer_dir  
    input_dir = os.path.join(args.input_dir, "env_data.pkl")
    output_dir = args.output_dir
    model = args.model

    export_sub_symbol(input_dir, cur_symbol_lst, model, output_dir)