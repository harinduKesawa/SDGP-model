from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re,string

#strings will print all the tweets within a dataset as a string
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

#before normalizing the tokens
# (tokanizing: converting a sentence to a array of individual words used module-punkt)
#print(tweet_tokens[0])

#method that normalizes the tokens
# def lemmatize_sentance(tokens):
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_sentance = []
#     #pos_tag: determins the context of the word
#     for word, tag in pos_tag(tokens):
#         if tag.startswith('NN'):
#             pos = 'n' #if the word is tagged with NN it'll be assigned as a noun
#         elif tag.startswith('VB'):
#             pos = 'v' #if the word is tagged with VB it'll be assigned as a verb
#         else:
#             pos = 'a' #else a adjective i think
#         lemmatized_sentance.append(lemmatizer.lemmatize(word,pos))
#     return lemmatized_sentance

#print(lemmatize_sentance(tweet_tokens[0]))

#this method both cleans, normalize and lemmatize the tweets
def remove_noice(tweet_tokens,stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith("VB"):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token,pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

from nltk.corpus import stopwords

stop_words = stopwords.words('english')

#print(remove_noice(tweet_tokens[0], stop_words))

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noice(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noice(tokens, stop_words))

#print(positive_tweet_tokens[500])
#print(positive_cleaned_tokens_list[500])

#
#DID NOT UNDERSTAND
#DID NOT UNDERSTAND
#DID NOT UNDERSTAND
#DID NOT UNDERSTAND
#DID NOT UNDERSTAND
#DID NOT UNDERSTAND
#
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

all_pos_words = get_all_words(positive_cleaned_tokens_list)

from nltk import FreqDist

freq_dist_pos = FreqDist(all_pos_words)
#print(freq_dist_pos.most_common(10))

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token,True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

import random

postive_dataset = [(tweet_dict, "positive")
                   for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "negative")
                    for tweet_dict in negative_tokens_for_model]

dataset = postive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]

from nltk import classify
from nltk import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_data)

#print("The accuracy is: ",classify.accuracy(classifier,test_data))

#print(classifier.show_most_informative_features(10))

from nltk.tokenize import word_tokenize

#custom_tweet = "One month ago I ordered one shoes of nike airmax from Flipkart it's cast was close to 7000 rs and it's bottom face was transparent . I thought it will be durable because it's cast was high in comparison with other brand. no doubt it's look is very nice but it is not durable , I could use my shoes only up to one month and now it is useless because of it is looking dull and bottom portion is gone. So I am suggesting you guys instead of taking this, local brand is better"
#custom_token = remove_noice(word_tokenize(custom_tweet))

#print(classifier.classify(dict([token, True] for token in custom_token)))

file = open('D:/New folder/comments.txt', "r")
for review in file:
    custom_review = review
    custom_token = remove_noice(word_tokenize(custom_review))
    print(classifier.classify(dict([token, True] for token in custom_token)))
