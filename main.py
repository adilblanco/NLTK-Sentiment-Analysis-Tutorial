import os
import nltk    

# set the download path
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# add path to NLTK
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('twitter_samples', download_dir=nltk_data_path)

    
# **Step 2**: Loading and Preprocessing Data
# For sentiment analysis, we need a labeled dataset of text and corresponding sentiment labels. NLTK provides the Twitter samples dataset:
from nltk.corpus import twitter_samples
import random

# Load twitter samples dataset
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Combine the datasets and create labels
tweets = positive_tweets + negative_tweets
labels = ['Positive'] * len(positive_tweets) + ['Negative'] * len(negative_tweets)

# Shuffle the dataset
combined = list(zip(tweets, labels))
random.shuffle(combined)
tweets, labels = zip(*combined)

# **Step 3**: Tokenization
# Tokenization is the process of splitting text into individual words or tokens. NLTK provides a simple way to tokenize text:
from nltk.tokenize import word_tokenize

sample_text = "NLTK is a powerful library for NLP."
tokens = word_tokenize(sample_text)
print(f"Step 3 - Tokenization :\n {tokens}\n___")

# **Step 4**: Removing Stopwords
# Stopwords are common words that do not carry significant meaning and can be removed from the text. NLTK has a built-in list of stopwords:
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]

filtered_tokens = remove_stopwords(tokens)
print(f"Step 4 - Removing Stopwords :\n {filtered_tokens}\n___")

# **Step 5**: Stemming and Lemmatization
# Stemming and lemmatization are techniques for reducing words to their root forms. NLTK provides tools for both:
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print(f"Step 5 - Stemming :\n {stemmed_tokens}\n___")
print(f"Step 5 - Lemmatization :\n {lemmatized_tokens}\n___")

# **Step 6**: Feature Extraction
# We need to convert our text data into a format suitable for machine learning algorithms. One common approach is to use a bag-of-words model:
from nltk.probability import FreqDist

all_words = [word.lower() for tweet in tweets for word in word_tokenize(tweet)]
all_words_freq = FreqDist(all_words)

# Select the top 2000 words as features
word_features = list(all_words_freq.keys())[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# Create feature sets for training and testing
feature_sets = [(document_features(word_tokenize(tweet)), label) for (tweet, label) in zip(tweets, labels)]
train_set, test_set = feature_sets[1000:], feature_sets[:1000]

# **Step 7**: Building a Sentiment Analysis Model
# We can use the Naive Bayes classifier, which is simple and effective for text classification tasks:
from nltk.classify import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(train_set)

# **Step 8**: Training and Evaluate the Model 
# To evaluate our model, we can use the accuracy metric:
import nltk.classify.util

accuracy = nltk.classify.util.accuracy(classifier, test_set)
print(f'Accuracy: {accuracy * 100:.2f}%')

# We can also display the most informative features:
classifier.show_most_informative_features(10)
