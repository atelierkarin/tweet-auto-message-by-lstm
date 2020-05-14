本サンプルは「ゼロから作るDeepLearning❷〜自然言語処理編〜（斎藤康毅 著）」を参考し、3層LSTMレイヤーを利用したRNNLMで自分のTwitterのつぶやきを学習し、文章を生成すること。
commonのコードは基本参考書とあまり変わりません（utils_jpだけは日本語読み込み用にアレンジ）。

つぶやきは日本語なので、janomeを利用した。さらにユーザー辞書で特殊な単語を登録し、単語化を改善する。

Tweetのソースファイルは提供しませんが、制作されたコーパスのPKLファイルは同梱されてる。いかのコードで必要な情報を取得できます。

```python
with open("tweet_corpus.pkl", 'rb') as f:
  data = pickle.load(f)
corpus, word_to_id, id_to_word = data
```

ちなみに学習完了のモデルもファイルサイズが大きすぎるため、ここではアップロードしません。

- *read_tweetjs.py* : Twitterからのデータの前作業、顔文字などを変換した（ソースは提供されてません）
- *convert_tweets_to_corpus.py* : read_tweetjsで出来たCSVを読み込み、それをコーパスなどを転換し、PKLファイルを保存
- *three_lstm_rnnlm.py* : 3層LSTMレイヤーを利用したRNNLMモデル
- *train.py* : 学習（簡略化のため検証データは作ってないのでその部分は省略）、学習完了のファイルはThreeLSTMRnnlm.pklとして保存
- *generate_text.py* : 文章生成、start_wordを変更すれば最初の言葉を変更できます