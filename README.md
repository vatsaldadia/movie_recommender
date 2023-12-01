# movie_recommender

Search
Write

Vatsaldadia
Movie Recommender using NLP(Word2Vec and TF-IDF)
Vatsaldadia
Vatsaldadia

7 min read
·
Just now






Photo by charlesdeluvio on Unsplash
We all love to watch movies, but there is always the question, ‘What should I watch next?’

So I have created a movie recommender which takes as input a movie title and gives the top 5 recommendations based on the overview of that particular movie.

I have used the MovieLens Database (9,000 titles) for this recommender.

To retrieve overviews for each title we will need the TMDB API. As part of the MovieLens Database, we get a CSV file called links.csv, which contains the corresponding tmdbId for each movieId, which is the identifier used across the MovieLens Database.

We start by importing all the necessary libraries.

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import gensim.downloader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
Using the links.csv we get the tmdbId

df1 = pd.read_csv('links.csv', usecols=['tmdbId'])  # just use tmdbId column
df1 = df1.dropna()  # remove null values
df1 = df1.convert_dtypes()  # convert float values to int values
We add two columns to the data frame, title and overview and assign them null values.

df1 = df1.assign(title = pd.NA, overview = pd.NA)
We make the API calls to get the titles and overviews. You can get your API authorizing token by signing up on the TMDB website.

for index, row in tqdm(df1.iterrows()):

    headers = {
        "accept": "application/json",
        "Authorization": "INSERT YOUR KEY"
    }

    url = "https://api.themoviedb.org/3/movie/" + str(row['tmdbId'])

    response = requests.get(url, headers = headers)
    result = response.json()
    try:
        df1.loc[index, 'title'] = result['original_title']
        df1.loc[index, 'overview'] = result['overview']
    except:
        continue

df1 = df1.dropna()
Now we have our overviews, but they can’t be used for processing as they are not ‘clean’, for which we have to remove stopwords, like ‘a’, ‘an, ‘my’, etc., as these words are not relevant for analyzing text semantics. We then tokenize the data that is split the words and store them in a list. Then we lemmatize all the words, i.e., convert all the words to their root form. The image below explains the key difference between stemming and lemmatization, the former just chops the word and the latter preserves context and gives better performance. We use the NLTK library for these pre-processing tasks.


A more detailed explanation can be found in this article, from where this image was taken.

Stemming vs Lemmatization in NLP
Words usually have multiple meanings based on their usage in the text. Similarly, different word forms convey related…
nirajbhoi.medium.com

We also need to remove any HTML tags as the text has been retrieved from an API. For this purpose, we use the BeautifulSoup library. We remove all the punctuations and non-alphanumeric characters using regex.

stop_words = set(stopwords.words('english'))

def clean(text):
    # Remove HTML tags and patterns
    clean_text = BeautifulSoup(text, "html.parser").get_text()

    # Convert text to lowercase
    clean_text = clean_text.lower()

    # Tokenize the text and remove stopwords
    words = nltk.word_tokenize(clean_text)
    words = [word for word in words if word not in stop_words]

    # Lemmatize words
    words = [WordNetLemmatizer.lemmatize(word) for word in words]

    # Remove punctuation and non-alphanumeric characters
    words = [word for word in words if re.match(r'^[a-zA-Z0-9]+$', word)]

    # Join the words back into a cleaned text
    cleaned_text = ' '.join(words)

    return cleaned_text
Run this function for each overview in our data frame, and delete the overview column as it is no longer required. The apply method from the Pandas library works similarly to the in-built Python map function.

df1['clean_overview'] = df1['overview'].apply(clean)
df1 = df1.drop(['overview'], axis=1)
We store the whole data frame in a CSV file

df1.to_csv("movie_data.csv", index=False)
Before we get into more code let’s understand Word2Vec and TF-IDF.

TF-IDF stands for term frequency-inverse document frequency. Term frequency calculates how often a term appears in a document (overview). The larger the TF the more times it has been used in the document. Inverse document frequency measures the importance of that term in the entire corpus (database). A higher IDF indicates that a term is less common across the corpus and, therefore, more important. The TF-IDF score is the product of TF and IDF for each term in each document. So a TF-IDF matrix has all the words in the corpus as columns and the document number as rows.


It is better than a count vectorizer because it gives preference to less common words, by giving them a higher score.


A detailed explanation of TF-IDF can be found here.

Understanding TF-IDF (Term Frequency-Inverse Document Frequency) - GeeksforGeeks
A Computer Science portal for geeks. It contains well-written, well thought and well-explained computer science and…
www.geeksforgeeks.org

Word embeddings are used in NLP to give numerical context to a word. They are the vector representations of words. Word2Vec is a word embedding technique. It has two architectures, CBOW (Continous Bag of Words) and Skip-Gram. More about them can be found in the below article. It predicts the probability of adjacent words based on the context it has learned by analyzing previous data.

Introduction to Word Embedding and Word2Vec
Word embedding is one of the most popular representations of document vocabulary. It is capable of capturing the context of…
towardsdatascience.com

Word2Vec as the name suggests represents a word in n-dimensional vector space. The values represent semantics and contextual information, so the higher the dimension, the better the word embeddings in terms of semantics. From the image below we can see length of the vector between king and man is almost equal to the length of the vector between queen and man, signifying a similar context.


Word2Vec embeddings reduced to 2-dimension
As the corpus used is very small training Word2Vec might not be the best approach, so we use a pre-trained model from Google which was trained on a part of Google News (about 100 billion words). The model contains 300-dimensional vectors for 3 million words and phrases. It can be loaded using the gensim library.

pretrained_model = gensim.downloader.load('word2vec-google-news-300')
We can also choose to fine-tune the model by re-training it on our corpus, but relative to the model size our corpus is small, so fine-tuning will not change the embeddings much.

We now run the TfidfVectorizer from the skicit-learn library on the corpus.

tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df = 2, stop_words='english')
tfidf_vectors = tfidf.fit_transform(movie_df['clean_overview'])
tfidf_feature = tfidf.get_feature_names_out()
vectors_df = pd.DataFrame(tfidf_vectors.toarray(), columns = tfidf_feature)
By specifying the min_df as 2 the vectorizer considers only terms which have occurred at least twice in the whole corpus. vectors_df is the data frame with a TF-IDF vector for each document.

We now want to calculate vectors for each document. To achieve this we take the weighted average of the individual word vectors belonging to that document. Using TF-IDF allows us to give importance to rare words as they are the determining words of an overview, and using Word2Vec helps preserve the semantic information of the data.

doc_vectors = []
for index, desc in tqdm(enumerate(corpus)):
    weighted_word_vector = np.zeros(300)
    weighted_sum = 0
    for word in desc:
        if word in tfidf_feature and word in pretrained_model:
            weighted_word_vector += pretrained_model[word] * vectors_df.loc[index, word]
            weighted_sum += vectors_df.loc[index, word]
        if weighted_sum != 0:
            weighted_word_vector /= weighted_sum
    doc_vectors.append(weighted_word_vector)
We then run cosine_similarity on the whole 2-dimensional doc_vectors array. Cosine similarity shows how close two given vectors are in the vector space concerning the angle between them.

cosine_similarities = cosine_similarity(doc_vectors, doc_vectors)
We finally write the function which gives us the recommendations. It takes a movie title as an argument, finds its index in the cosine_similarity matrix, and maps the indices of the top 5 similar vectors to their corresponding movie titles.

def recommendation(title):
    indices = pd.Series(movie_df.index, index = movie_df['title']).drop_duplicates()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:6]
    print(sim_scores)
    book_indices = [i[0] for i in sim_scores]
    recommend = movie_df.iloc[book_indices]["title"]
    return recommend
We test our code by asking for recommendations for the movie Mission Impossible

recommendation("Mission: Impossible")
7782                     Goon
7267               The Losers
4087                    Thief
7766    A Very Potter Musical
2293             Meatballs IV
Name: title, dtype: object
These were the movies recommended by the code. When we check the overviews of each movie and compare them to the overview of the movie Mission Impossible from the TMDB website, we can notice the similarity.

We can also opt to print the cosine similarity scores for the recommended movies.

[(7778, 0.9987755673810671), (7265, 0.994846548056306), (4086, 0.9850812766645703), (7762, 0.9835303305505907), (2292, 0.969345692097803)]
If we had used unweighted averages for word2vec vectors while calculating the document vectors, i.e., without calculating the TF-IDF vectors, for Mission Impossible, the recommender would have given the next movies in the franchise, i.e, Mission Impossible II, Mission Impossible III, etc., which the user already would have already seen/planning to see, making this engine futile. So at least for this given dataset TF-IDF weighted averages of Word2Vec vectors seem to work better compared to the unweighted averages. The dataset is very small, better results and recommendations could be achieved if we used a larger dataset.

This engine only recommends movies which are in the dataset as only they have their cosine similarity scores calculated.

The code can be expanded to accommodate new movies by mapping new titles to indices, cleaning the overview of that movie, calculating the weighted average for that overview, and then adding a row to the cosine_similarity matrix and calculating the score for each column, i.e., each movie overview.

Thank you for reading.
