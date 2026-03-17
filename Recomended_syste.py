import pandas as pd 
import numpy as np 
from  sklearn.model_selection  import train_test_split

from sklearn.pipeline import make_pipeline 
import seaborn as sns 

import ast

import nltk 

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


df_movies = pd.read_csv('/Users/tawhidurrahman/ML_projects/Movie_recommended_system/Movie_database/tmdb_5000_credits.csv')
cradit = pd.read_csv('/Users/tawhidurrahman/ML_projects/Movie_recommended_system/Movie_database/tmdb_5000_movies.csv')



df_movies = df_movies.merge(cradit, on='title')


df_movies = df_movies[["movie_id", 'title' , 'overview' , 'genres' , 'keywords' , 'cast' , 'crew']]
df_movies = df_movies.dropna()


def convert (object): 
    
    list = []
    
    for i in ast.literal_eval(object):
        list.append(i['name'])
    return list



df_movies['genres'] = df_movies['genres'].apply(convert)

df_movies['keywords'] = df_movies['keywords'].apply(convert)


def convert_cast(obj):
    list1 = []
    count = 0
    for i in ast.literal_eval(obj):
        if count != 3:
            list1.append(i['name'])
            count += 1
        else:
            break
    return list1

df_movies['cast'] = df_movies['cast'].apply(convert_cast)


def find_director (object): 
    
    list2 = []
    
    for i in ast.literal_eval(object):
        if i['job'] == 'Director' :
            list2.append(i['name'])
            break 
    return list2

df_movies['crew'] = df_movies['crew'].apply(find_director)


df_movies['overview'] = df_movies['overview'].apply(lambda x:x.split())



df_movies['genres'] =  df_movies['genres'].apply (lambda x : [i.replace(" " , "")for i in x])

df_movies['keywords'] =  df_movies['keywords'].apply (lambda x : [i.replace(" " , "")for i in x])

df_movies['cast'] =  df_movies['cast'].apply (lambda x : [i.replace(" " , "")for i in x])

df_movies['crew'] =  df_movies['crew'].apply (lambda x : [i.replace(" " , "")for i in x])

df_movies['tags'] = df_movies['overview'] + df_movies['genres'] + df_movies['keywords'] + df_movies['cast'] + df_movies['crew']



new_df = df_movies[['movie_id', 'title', 'tags']].copy()

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


new_df['tags'] = new_df['tags'].apply(stem)



from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features= 5000 , stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()
print(vectors)

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

def recommened(movie) :
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)) , reverse=True , key= lambda x:x[1])[1:6]
    for i in movies_list :
        print(new_df["title"][i[0]])
        

import pickle 


pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))




