import pandas as pd
import re
# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Setup global 

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
# TEXT CLENAING
TEXT_CLEANING_RE = re.compile("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+")


def preprocess(text, stem=False):
    """
        This method will process individual tweets and get rid of stopwords,
        usernames, and other things that are not needed.
    """
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    # Get rid of stop words
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


def clean_tweet(tweet): 
    """
    Utility function to clean tweet text by removing links, special characters 
    using simple regex statements. 
    """
    return ' '.join(re.sub(TEXT_CLEANING_RE, " ", tweet).split()).lower()


def clean_tweets(file_path):
    """
        This method will clean the tweets and store them in a DataFrame.
        (Assumes header has been removed)

        Args:
            file_path (str): path to tweets text file
    """
    # Regex for unique rows
    regx = re.compile('([0-9]+\s+[+-]\s+)')
    # Open and read text file
    with open(file_path, 'r') as f:
        headers = f.readline()
        headers = re.split('\s+|\\n', headers)[:-1]
        text = f.read()
    # Split string by regex
    string_list = re.split(regx, text)[1:]
    # Seperate and join into single list of lists
    l1 = [re.split('\s+', i)[:2] for i in string_list[0::2]]
    l2 = string_list[1::2]
    for i in range(len(l1)):
        l1[i].append(l2[i])
    # Cast list of lists to DataFrame
    df = pd.DataFrame(l1, columns=headers)
    # Replace + with 1 and - with 0
    df['Sentiment1'] = df['Sentiment1'].apply(
        lambda x: 1 if x == '+' else 0)
    df['SentimentText'] = df['SentimentText'].apply(preprocess)#clean_tweet)
    # Rename columns
    df.rename(columns={'TonyID': 'id', 'Sentiment1': 'target',
                       'SentimentText': 'text'},
              inplace=True)

    return df

if __name__ == '__main__':
    # Test clean_tweets function
    file_name = 'Training.txt'
    df = clean_tweets(file_name)
    print(df.head())