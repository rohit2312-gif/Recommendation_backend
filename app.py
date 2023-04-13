import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')
tmdb=pd.read_csv('tmdb.csv')


# In[3]:


a=2
print(a)


# In[ ]:


movies.head()


# In[ ]:


credits.head()


# In[ ]:


credits.columns = ['id','title','cast','crew']

movies = movies.merge(credits, on="id")


# In[ ]:


movies.head()


# In[ ]:


from ast import literal_eval

features = ["cast", "crew", "keywords", "genres"]

for feature in features:
    movies[feature] = movies[feature].apply(literal_eval)

movies[features].head(10)


# In[ ]:


import numpy as np
def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan


# In[ ]:


def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]

        if len(names) > 3:
            names = names[:3]

        return names

    return []


# In[ ]:


movies["director"] = movies["crew"].apply(get_director)

features = ["cast", "keywords", "genres"]
for feature in features:
    movies[feature] = movies[feature].apply(get_list)


# In[ ]:


movies[['original_title', 'cast', 'director', 'keywords', 'genres','id']].head()


# In[ ]:


def clean_data(row):
    if isinstance(row, list):
        return [str.lower(i.replace(" ", "")) for i in row]
    else:
        if isinstance(row, str):
            return str.lower(row.replace(" ", ""))
        else:
            return ""

features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    movies[feature] = movies[feature].apply(clean_data)


# In[ ]:


def create_soup(features):
    return ' '.join(features['keywords']) + ' ' + ' '.join(features['cast']) + ' ' + features['director'] + ' ' + ' '.join(features['genres'])


movies["soup"] = movies.apply(create_soup, axis=1)


from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(movies["soup"])

print(count_matrix.shape)

cosine_sim2 = cosine_similarity(count_matrix, count_matrix) 
print(cosine_sim2.shape)

movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies['original_title'])


# In[ ]:


indices = pd.Series(movies.index, index=movies["original_title"]).drop_duplicates()

print(indices.head())


# In[ ]:





# In[ ]:


def get_recommendations(title, cosine_sim=cosine_sim2):
    idx = indices[title]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores= sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores= similarity_scores[1:11]
    # (a, b) where a is id of movie, b is similarity_scores

    movies_indices = [ind[0] for ind in similarity_scores]
    recommends = movies["id"].iloc[movies_indices]
    
    return recommends


# In[ ]:


ans=get_recommendations('Toy Story',cosine_sim2)


# In[ ]:


for a in ans:
    #https://api.themoviedb.org/3/movie/49026?api_key=389b8a0210ce66880d4f3b0127b4b9ae&language=en-US#
    print(f'https://api.themoviedb.org/3/movie/{a}?api_key=389b8a0210ce66880d4f3b0127b4b9ae&language=en-US')


# In[ ]:


ans


# In[ ]:


from flask import Flask,request,jsonify
from flask_cors import CORS
import json


# In[ ]:


app = Flask(__name__)
 
CORS(app)        
@app.route('/',methods=['GET'])

def recommend_movies():
    arg=request.args
    movie=arg.get("name")
    
    response = get_recommendations(movie,cosine_sim2)
    json_response=[]
    for res in response:
        json_response.append(f'https://api.themoviedb.org/3/movie/{res}?api_key=389b8a0210ce66880d4f3b0127b4b9ae&language=en-US')
        
    return jsonify({"Results": json_response})

   

if __name__=='__main__':
        app.run(debug = True,use_reloader=False)


# In[ ]:




