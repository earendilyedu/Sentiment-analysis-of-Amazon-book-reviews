"""
This file reads in the following Yelp datasets:

- review data set
- business data set
- user data set

It then merges the three datasets and retains only Restaurants
that have a large number of reviews. It then writes the resulting
DataFrame to disk for later consumption
"""

import json
import pandas as pd
import os
from sklearn.cross_validation import train_test_split
from collections import defaultdict
import sys
sys.path.append('/Users/Louis/final-project/classes/main')
from bus.review import Review

#### function to define train data sentiment
def partition(x):
    if x < 3:
        return 'Negative'
    if x == 3:
        return 'Objective'
    return 'Positive'


def load_file(path):
    ### read file as dataframe
    with open(path) as f:
        df = pd.DataFrame([json.loads(r) for r in f.readlines()])





###### keep columns we want
    keeps =['asin', 'helpful', 'overall', 'reviewText', 'reviewerID']
    final_df = df[keeps]


# FEATURIZE




    list_of_list_sents = [Review(review,5).sentence_tokenize() for review in final_df.reviewText]

    for sents,star in zip(list_of_list_sents,final_df.overall):
        for sent in sents:
            sent.stars=star # pass the number of stars in

    ### get final featurized_df
    lst = []
    for sent_list in list_of_list_sents:
        d = defaultdict(int)
        for sent in sent_list:
            features = sent.get_features()
            for feature in features:
                if feature != 'review_stars':
                    d[feature] += features[feature]
                else:
                    d[feature] = features[feature]
        lst.append(d)
    ##### convert lst to dataframe
    featurized_df = pd.DataFrame(lst)

    featurized_df['sentiment'] = featurized_df['review_stars'].map(partition)
    featurized_df = featurized_df[~featurized_df.sentiment.isnull()]




    # Adjust sentiment labels
    featurized_df.sentiment[featurized_df.sentiment=='Positive'] = 1
    featurized_df.sentiment[featurized_df.sentiment=='Negative'] = -1
    featurized_df.sentiment[featurized_df.sentiment=='Objective'] = 0

    # Create 'opinionated' variable
    featurized_df['opinionated'] = 0
    featurized_df['opinionated'][featurized_df.sentiment!= 0] = 1



    ######## store our preprocessed data

    featurized_df.to_csv("/Users/Louis/final-project/data/featurized_train.csv", index=False,header=True)

    print "Done."



if __name__ == "__main__":
    path = '/Users/Louis/final-project/data/Book_reviews50000.json'
    load_file(path)
