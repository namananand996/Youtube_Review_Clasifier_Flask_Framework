import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB

dataset = pd.read_csv("Restaurant_Reviews.tsv" , delimiter='\t' , quoting = 3)
nltk.download('stopwords')
corpus = []

for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set (stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

countvectorizer = CountVectorizer()
X = countvectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)


clasifier = MultinomialNB()
#clasifier = GaussianNB()
clasifier.fit(X_train,y_train)
clasifier.score(X_test,y_test)



namequery = "Food was not that much up to mark"
data = [namequery]
vect = countvectorizer.transform(data).toarray()
my_prediction = clasifier.predict(vect)




