import sys
sys.path.append("..")

import pickle

from common import config
config.GPU = True
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from three_lstm_rnnlm import ThreeLSTMRnnlm

with open("tweet_corpus.pkl", 'rb') as f:
  data = pickle.load(f)
corpus, word_to_id, id_to_word = data

# ハイパーパラメータを設定
batch_size = 20
wordvec_size = 800
hidden_size = 800
time_size = 35
lr = 20.0
max_epoch = 20
max_grad = 0.25
dropout = 0.5

vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

model = ThreeLSTMRnnlm(vocab_size, wordvec_size, hidden_size, dropout)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# 学習
for epoch in range(max_epoch):
  trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size, time_size=time_size, max_grad=max_grad, eval_interval=20)
  model.save_params()
  model.reset_state()
  print('-' * 50)

# パラメータを保存
model.save_params()