# -*- coding: utf-8 -*-
# エクセルを立ち上げておいてファイル読み込む
import json
from dateutil.parser import parse
from pytz import timezone
import re
import sys
import io
import codecs

# jasonファイルを開いて読み込む
with codecs.open('tweet.js', 'r' , 'utf-8', 'ignore') as f:
    data = f.read()
tw = json.loads(data[data.find('['):])

# 書き出すファイルを開く
f = open('tweet.csv', 'wb')

# 見たいデータを出力する
for t in tw:
  s = t['tweet']['full_text']

  # リツイートはスキップ
  if s[:2] == "RT":
    pass
  else:
    s = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', s)
    s = re.sub(r'@\S*\s?', '', s)
    s = re.sub("#\S*\s?","", s)
    s = re.sub("\(´・ω・｀\)","ｼｮﾎﾞｰﾝ", s)
    s = re.sub("\(´\,\,・ω・\,\,｀\)","ｼｮﾎﾞｰﾝ", s)    
    s = re.sub("\(´ーωー｀\)","ﾅﾅﾒ", s)
    s = re.sub("\(｀・ω・´\)","ｼｬｷｰﾝ", s)
    s = re.sub("\( ﾟдﾟ\)","ﾎﾟｶｰﾝ", s)
    s = re.sub("\( ﾟдﾟ\)","ﾎﾟｶｰﾝ", s)
    s = re.sub("\(´；ω；｀\)","ﾅﾐﾀﾞ", s)
    s = re.sub("\(=´・ω・`= \)","みっしぃ", s)
    s = re.sub("ヽ\(･ω･\)/","", s)

    # 除草作業
    s = re.sub("ww","ﾜﾗ", s)
    s = re.sub("ｗｗ","ﾜﾗ", s)
    s = re.sub("ﾜﾗw","ﾜﾗ", s)
    s = re.sub("ﾜﾗｗ","ﾜﾗ", s)
    s = re.sub("ﾜﾗﾜﾗ","ﾜﾗ", s)
    s = re.sub("ｗ\n","ﾜﾗ\n", s)

    s = re.sub("\.\.\.","…", s)
    s = re.sub("。。。","…", s)
    s = re.sub("…。","…", s)

    s = re.sub("\(ry","略", s)

    s = s.strip()
    s = s.replace("\n", " ")
    s = s.replace("\r", " ")
    if s != "":
      s = s + ' ｴﾝﾄﾞ\n'
      f.write(s.encode('utf-8'))
f.close()