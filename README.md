# predict_num_likes
web_crawler.py can download reviews for videos with top popularities in certain regions on Bilibili, and save them as .csv.

reply_wordcloud.py generates wordcloud based on reviews in a certain csv file. (column holding the review is named "content").

get_model.py tries to generate a model based on neural network to predict whether a review is gonna get high number of likes (num_like > 100).

analyze_length.py outputs the distribution of length of reviews in a certain dataset.

So far, it is clear that model's precision and recall for high-like reviews cannot surpass 20% unless more features other than the text themselves can be extracted from the dataset. This is room for more work to be done.
