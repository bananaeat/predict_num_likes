import pandas as pd
import jieba

replies_dataframe = pd.read_csv("./saved/waimai_10k.csv")
replies = replies_dataframe["review"]
replies_length = pd.Series([len(jieba.lcut(x)) for x in replies]).sort_values()
print(replies_length)

