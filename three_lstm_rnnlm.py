import sys
sys.path.append("..")
from common.time_layers import *
from common.base_model import BaseModel

class ThreeLSTMRnnlm(BaseModel):
  def __init__(self, vocab_size=10000, wordvec_size=650, hidden_size=650, dropout_ratio=0.5):
    V, D, H = vocab_size, wordvec_size, hidden_size

    embed_W = (0.01 * np.random.randn(V, D)).astype("f")

    lstm_Wx1 = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype("f")
    lstm_Wh1 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
    lstm_b1 = np.zeros(4 * H).astype("f")

    lstm_Wx2 = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype("f")
    lstm_Wh2 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
    lstm_b2 = np.zeros(4 * H).astype("f")

    lstm_Wx3 = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype("f")
    lstm_Wh3 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
    lstm_b3 = np.zeros(4 * H).astype("f")

    affine_b = np.zeros(V).astype("f")

    self.layers = [
      TimeEmbedding(embed_W),
      TimeDropout(dropout_ratio),
      TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1),
      TimeDropout(dropout_ratio),
      TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2),
      TimeDropout(dropout_ratio),
      TimeLSTM(lstm_Wx3, lstm_Wh3, lstm_b3),
      TimeDropout(dropout_ratio),
      TimeAffine(embed_W.T, affine_b)
    ]
    self.loss_layer = TimeSoftmaxWithLoss()
    self.lstm_layers = [self.layers[x] for x in [2,4,6]]
    self.drop_layers = [self.layers[x] for x in [1,3,5,7]]

    self.params, self.grads = [], []
    for layer in self.layers:
      self.params += layer.params
      self.grads += layer.grads
  
  # 学習中ではDropoutを有効にします、それ以外では無効にします
  def predict(self, xs, train_flg=False):
    for layer in self.drop_layers:
      layer.train_flg = train_flg
    for layer in self.layers:
      xs = layer.forward(xs)
    return xs
  
  def forward(self, xs, ts, train_flg=True):
    score = self.predict(xs, train_flg)
    loss = self.loss_layer.forward(score, ts)
    return loss
  
  def backward(self, dout=1):
    dout = self.loss_layer.backward(dout)
    for layer in reversed(self.layers):
      dout = layer.backward(dout)
    return dout
  
  def reset_state(self):
    for layer in self.lstm_layers:
      layer.reset_state()