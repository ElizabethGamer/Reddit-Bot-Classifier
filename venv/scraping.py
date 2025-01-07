import pandas as pd
from scipy.sparse import csr_matrix
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# data = pd.read_csv("./bot_detection_data.csv")

# array of tweets, array of account types
data = pd.read_csv("twitter_human_bots_dataset.csv")
data = data.filter(items=["description", "lang", "account_type"])
data = data[data['lang'] == "en"]

tweets = data.filter(items=["description"])
# stemming
tweets = tweets.to_numpy()
stemmer = SnowballStemmer('english')

tweets = [' '.join([stemmer.stem(word) for word in text[0].split(' ')]) for text in tweets]

accounts = data.filter(items=["account_type"])
accounts = accounts.replace("human", value="0")
accounts = accounts.replace("bot", value="1")
accounts = accounts.astype("int32")
accounts = accounts.to_numpy()
accounts = accounts[:, 0]
print(accounts.shape)

classifiers = [ExtraTreesClassifier()]

for classifier in classifiers:
    print(classifier)

    # vect = CountVectorizer(lowercase=True, strip_accents='ascii', stop_words='english')
    # vect_twts = pd.DataFrame()
    # for index, row in stemmed_twts.iterrows():
    #     try:
    #         newrow = vect.fit_transform(row.iloc[0].split(' '))
    #         newrow = pd.DataFrame({'description': newrow}, index=[0])
    #         vect_twts = pd.concat([vect_twts, newrow])
    #     except Exception as e:
    #         newrow = csr_matrix((1, 1), dtype="int32")
    #         newrow = pd.DataFrame({"description": newrow}, index=[0])
    #         vect_twts = pd.concat([vect_twts, newrow])
    #
    #
    # tfidf = TfidfTransformer()
    # tf_twts = pd.DataFrame()
    # for index, row in vect_twts.iterrows():
    #     newrow = tfidf.fit_transform(row.iloc[0])
    #     newrow = pd.DataFrame({"description": newrow}, index=[0])
    #     tf_twts = pd.concat([tf_twts, newrow])

    text_clf = Pipeline([
		('vect', CountVectorizer(lowercase=True, strip_accents='ascii', stop_words='english')),
		('tfidf', TfidfTransformer()),
        ('clf', classifier),
    ])


    twt_train, twt_test, acct_train, acct_test = train_test_split(tweets, accounts, test_size=0.10, random_state=31)

    # Train model
    # classifier.fit(list(twt_train), acct_train)
    text_clf.fit(twt_train, acct_train)

    # Predict test data
    acct_pred = text_clf.predict(list(twt_test))
    print(confusion_matrix(acct_pred, acct_test))
    print('accuracy: ' + str(accuracy_score(acct_test, acct_pred)))
    print(classification_report(acct_pred, acct_test))


