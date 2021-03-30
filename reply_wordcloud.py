import pandas as pd
import matplotlib.pyplot as plt
import jieba
import jieba.posseg as pseg
from itertools import chain
from wordcloud import WordCloud


def get_a_list(text):
    r = []
    for g in pseg.lcut(text):
        if g.flag == "n":
            r.append(g.word)
    return r


data = pd.read_csv("./saved/replies.csv")
replies = data["content"]
vocab = list(chain(*map(lambda x: get_a_list(x), replies)))


def get_word_cloud(keywords_list):
    wordcloud = WordCloud(font_path="C:\\WINDOWS\\FONTS\\SIMSUN.TTC", max_words=100, background_color="white")
    keywords_string = " ".join(keywords_list)
    wordcloud.generate(keywords_string)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


get_word_cloud(vocab)
