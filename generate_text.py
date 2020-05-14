import sys
sys.path.append("..")

from rnnlm_gen import ThreeLSTMRnnlmGen

import pickle

with open("tweet_corpus.pkl", 'rb') as f:
  data = pickle.load(f)
corpus, word_to_id, id_to_word = data

vocab_size = len(word_to_id)
corpus_size = len(corpus)
wordvec_size = 800
hidden_size = 800

model = ThreeLSTMRnnlmGen(vocab_size, wordvec_size, hidden_size)
model.load_params("ThreeLSTMRnnlm.pkl")

start_word = "今日"
start_id = word_to_id[start_word]
skip_words = [" "]
skip_ids = [word_to_id[w] for w in skip_words]

word_ids = model.generate(start_id, skip_ids, 50)
txt = "".join([id_to_word[i] for i in word_ids])
txt = txt.replace("ｴﾝﾄﾞ", "\n")
txt = txt.replace("ｼｮﾎﾞｰﾝ", "(´・ω・｀)")
txt = txt.replace("ﾜﾗ", "ｗ")
print(txt)