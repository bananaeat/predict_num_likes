import requests
import pandas as pd
import json

rply = []
regions = [172]
for region in regions:
    url = "https://api.bilibili.com/x/web-interface/ranking/region?rid=" + str(region) + "&day=7&original=0&jsonp=jsonp"
    response = requests.get(url=url)
    text = response.text
    ob = json.loads(text)
    videos = ob["data"]
    bvid = []
    aid = []
    for video in videos:
        bvid.append(video["bvid"])
        aid.append(video["aid"])

    for v in range(1, 11):
        for i in range(1, 101):
            url = "https://api.bilibili.com/x/v2/reply?jsonp=jsonp&pn=" + str(i) + "&type=1&oid=" + aid[v] + "&sort=2"
            response = requests.get(url=url)
            text = response.text
            ob = json.loads(text)
            replies = ob["data"]["replies"]
            if replies is None:
                break
            for reply in replies:
                reply_line = [reply["content"]["message"], reply["like"]]
                rply.append(reply_line)
            if i % 5 == 0:
                print("Scanned " + str((v-1) * 100 + i) + " pages")


data_rply = pd.DataFrame(data=rply, columns=["content", "num_like"])
data_rply.to_csv("./saved/replies_172.csv", encoding="utf_8_sig")