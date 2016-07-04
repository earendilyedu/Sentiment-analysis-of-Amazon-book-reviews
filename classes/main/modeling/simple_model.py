from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import classification_report, roc_curve, auc,accuracy_score
import pylab as pl
import pandas as pd
import pickle

# for SENTIMENT MODEL
## LogisticRegression

final_df = pd.read_csv("/Users/Louis/final-project/data/featurized_train.csv")

final_df_sent  = final_df[final_df.sentiment!=0].copy()
final_df_sent = final_df_sent.dropna(axis=0)

y = final_df_sent.sentiment.astype(int)
X = final_df_sent.drop(['sentiment', 'opinionated','review_stars'], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


clf = LogisticRegression(class_weight='balanced')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print 'Logistic'
print accuracy_score( y_test, y_pred)

## RandomForestClassifier

rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print 'random'
print accuracy_score(y_test, y_pred), classification_report(y_test, y_pred)



#### GradientBoostingClassifier
gb = RandomForestClassifier(class_weight='balanced')
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
print 'Boosting'
print accuracy_score( y_test, y_pred)


# for opinion MODEL
## LogisticRegression

final_df = pd.read_csv("/Users/Louis/final-project/data/featurized_train.csv")
final_df = final_df.dropna(axis=0)
y = final_df.opinionated.astype(int)
X = final_df.drop(['sentiment', 'opinionated','review_stars'], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


clf = LogisticRegression(class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
classification_report( y_test, y_pred)
print accuracy_score( y_test, y_pred)
## RandomForestClassifier

rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
classification_report( y_test, y_pred)
print accuracy_score( y_test, y_pred)


#### GradientBoostingClassifier
gb = RandomForestClassifier(class_weight='balanced')
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
classification_report( y_test, y_pred)




if __name__ == "__main__":
    print 'Done'
     # raw data set

    # with open('/Users/Louis/final-project/classes/main/models/opin_model.pkl', 'wb') as f:
	# 	pickle.dump(opin_best_est, f)
	# with open('/Users/Louis/final-project/classes/main/models/senti_model.pkl', 'wb') as f:
	# 	pickle.dump(senti_best_est, f)
