from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
import random
import nltk
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes));
        conf = float(choice_votes) / len(votes)
        return conf
        
        
        


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories() 
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

all_words = [w.lower() for w in movie_reviews.words()]
all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        
    return features
 
#print (find_features(movie_reviews.words('neg/cv000_29416.txt')))
featuresets = [(find_features(rev), category) for (rev, category) in documents]
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print 'Naive Bayes Algo accuracy percent:', (nltk.classify.accuracy(classifier,testing_set)*100)
classifier.show_most_informative_features()

MNB_Classifier = SklearnClassifier(MultinomialNB())
MNB_Classifier.train(training_set)
print 'MNB_Classifier : Algo accuracy percent:', (nltk.classify.accuracy(MNB_Classifier,testing_set)*100)
    
    
#GaussianNB_Classifier = SklearnClassifier(GaussianNB())
#GaussianNB_Classifier.train(training_set)
#print 'GaussianNB_Classifier : Algo accuracy percent:', (nltk.classify.accuracy(GaussianNB_Classifier,testing_set.toarray())*100)

BernoulliNB_Classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_Classifier.train(training_set)
print 'BernoulliNB_Classifier : Algo accuracy percent:', (nltk.classify.accuracy(BernoulliNB_Classifier,testing_set)*100)  

LogisticRegression_Classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_Classifier.train(training_set)
print 'LogisticRegression_Classifier : Algo accuracy percent:', (nltk.classify.accuracy(LogisticRegression_Classifier,testing_set)*100) 



SGDClassifier_Classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_Classifier.train(training_set)
print 'SGDClassifier_Classifier : Algo accuracy percent:', (nltk.classify.accuracy(SGDClassifier_Classifier,testing_set)*100) 


#SVC_Classifier = SklearnClassifier(SVC())
#SVC_Classifier.train(training_set)
#print 'SVC_Classifier : Algo accuracy percent:', (nltk.classify.accuracy(SVC_Classifier,testing_set)*100) 


LinearSVC_Classifier = SklearnClassifier(LinearSVC())
LinearSVC_Classifier.train(training_set)
print 'LinearSVC_Classifier : Algo accuracy percent:', (nltk.classify.accuracy(LinearSVC_Classifier,testing_set)*100) 


NuSVC_Classifier = SklearnClassifier(NuSVC())
NuSVC_Classifier.train(training_set)
print 'NuSVC_Classifier : Algo accuracy percent:', (nltk.classify.accuracy(NuSVC_Classifier,testing_set)*100) 



voted_classifier = VoteClassifier(classifier,MNB_Classifier, BernoulliNB_Classifier, LogisticRegression_Classifier, SGDClassifier_Classifier, LinearSVC_Classifier, NuSVC_Classifier)
print 'voted_classifier : Algo accuracy percent:', (nltk.classify.accuracy(voted_classifier,testing_set)*100) 

print 'classification: ', voted_classifier.classify(testing_set[0][0]), 'Confidence:' , voted_classifier.confidence(training_set[0][0])
