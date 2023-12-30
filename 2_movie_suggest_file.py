import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

df = pd.read_csv('tmdb_5000_movies.csv')
pd.set_option('display.max_columns', None)
# print(df.head())

#To access the data format
x = df.iloc[0]
print(x['genres'], end='\n\n\n') #Found out it is in json format
print(x['keywords']) #Found out it is in json format
print(x.dtypes)

def combine_keyword(row):
    genres = json.loads(row['genres'])
    genres = ' '.join(''.join(each_dic['name'].split()) for each_dic in genres)

    keywords = json.loads(row['keywords'])
    keywords = ' '.join(''.join(each_dic['name'].split()) for each_dic in keywords)
    return "%s %s" % (genres, keywords)


#create a new string representation of each movie
df['combined_string'] = df.apply(combine_keyword, axis=1)
print(df['combined_string'])

#Create a Tf-idf vectorizer object (3000 here limit the number of column, only more commonly used words is included)
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)

#create a data matrix from the df['combined string']
string_matrix = tfidf.fit_transform(df['combined_string'])
print(string_matrix)

#########################################################################
#generate a new mapping series called "movie2idx" which uses the movie title as index, the original index will become the value now (like switching places.)
movie2idx = pd.Series(df.index, index=df['title'])
# print(movie2idx)

idx = movie2idx['Street Fighter: The Legend of Chun-Li']
# print(idx)

#extract the search vector
search = string_matrix[idx]
search.toarray()

#compute the similiarity between search and every vector in it
scores = cosine_similarity(search, string_matrix)
scores = scores.flatten() #score is the column of movie with the similiary score beside it
plt.plot(scores)
plt.show()
# print(search)

###########################################################
print('After Argsort array', (-scores).argsort())
# Argsort is will just return the actual score back to the 'numpy.ndarray' in the form of ranking
# In arg sort, the lower number will get higher ranking/index
# and hence -ve sign is put in front to put the ones with higher magnitude in the top rank

plt.plot(scores[(-scores).argsort()])
plt.show()
# Both scores and (-scores).argsort() have to be a  numpy index array for it to work

#get to 5 matches, and remember to exclude oneself
recommended_idx = (-scores).argsort()[1:6]

#convert the recommended index back to the title
df['title'].iloc[recommended_idx]

# create a function that generates movie suggestions
def suggest(title):
  # get the row in the dataframe for this movie
  idx = movie2idx[title]     #Normally we put in the index we can find the content, but since we have inverted than, so we are able to put title and find back the index
  if type(idx) == pd.Series: #Due to the inconsistency of pandas API, there may be a few same title, and pandas might return more than 1 idx
    idx = idx.iloc[0]        #Hence in here we choose to select the first row

  # calculate the pairwise similarities for this movie
  search = string_matrix[idx]
  scores = cosine_similarity(search, string_matrix)

  #Now the array is 1 x N, let's just make it just a 1-D array directly
  scores = scores.flatten()

  # get the indexes of the highest scoring movies
  # get the first K recommendations
  # don't return itself!
  recommended_idx = (-scores).argsort()[1:6]

  # return the titles of the recommendations
  return df['title'].iloc[recommended_idx]

print("Recommendations for 'DOA: Dead or Alive':")
print(suggest('DOA: Dead or Alive'))

print("Recommendations for 'The Calling':")
print(suggest('The Calling'))

print("Recommendations for 'Me, Myself & Irene':")
print(suggest('Me, Myself & Irene'))
