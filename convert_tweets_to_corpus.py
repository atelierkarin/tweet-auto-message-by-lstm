import sys
import os
sys.path.append("..")
from common.utils_jp import preprocess

import pickle

text = ""
with open('tweet.csv', encoding='utf-8') as f:
  curr_text = f.read()
  if curr_text != "":
    text += curr_text

custom_dict = os.getcwd() + "\\user_simpledic.csv"

corpus, word_to_id, id_to_word = preprocess(text, custom_dict)

data = (corpus, word_to_id, id_to_word)

print(word_to_id)

with open("tweet_corpus.pkl", 'wb') as f:
  pickle.dump(data, f)