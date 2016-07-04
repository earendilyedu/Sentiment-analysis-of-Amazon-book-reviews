import json
import pandas as pd

import sys
sys.path.append('/Users/Louis/final-project/classes/main/bus')
from business import Business


with open(path) as f:
    reviews_df = pd.DataFrame([json.loads(r) for r in f.readlines()])


bus_dict = {}
for i in reviews_df.asin.unique():
    index = i
    bus_dict[index] = df[df.asin == index][['reviewText','overall']]

Final_result ={}
for key in bus_dict.keys():
    Final_result[key] = Business(bus_dict,key)

for i in Final_result.iteritems():
    i.extract_aspects()
    # i.aspect_summary()
    i.aspect_based_summary()



if __name__ == "__main__":
    print done
