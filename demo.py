#!/usr/bin/env python
# coding: utf-8

# In[48]:


# import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import ast
a=ast.literal_eval 
#list within a string is displayed
#We start by performing Exploratory Data Analysis on given datset. We use two kaggle datasets for this particular project.

mov = pd.read_csv('tmdb_5000_movies.csv', index_col='id')

movie_list=mov["original_title"]
mov['original_language'].unique()
mov['genres']
creds = pd.read_csv('tmdb_5000_credits.csv',index_col='movie_id')
creds.head()
movie_list
creds.info()
creds.shape
mov.info()
mov.describe()
mov.shape
creds.describe()
#We concatenate the two dataframes into a single one, by removing title column which appears twice in the final dataframe and might intervene with the results (two columns with same name).

new_df = pd.concat([mov,creds],axis=1)
new_df.shape
new_df = new_df.loc[:,~new_df.columns.duplicated()].copy()
print(mov.loc[19995])
print('\n')
print(creds.loc[19995])
print('\n')
print(new_df.loc[19995])
new_df.columns
#Now we perform data cleaning by either removing the unnecessary values or imputing them as per requirement.

new_df.isnull().sum()
new_df.drop(['homepage','tagline'],inplace=True,axis=1)
new_df.isnull().sum()
new_df['overview'] = new_df['overview'].fillna("Unknown")
print(new_df['runtime'].mean())
new_df['runtime'] = new_df['runtime'].fillna(new_df['runtime'].mean())
new_df['release_date'] = new_df['release_date'].fillna("Unknown")
new_df.isnull().sum()
new_df.duplicated().sum()
#Since data in the given "genres" column is in the form of a string object, we use ast.literal_evals. This helper function evaluates an expression node or a string consisting of a Python literal or container display. Thus the type of ast.literal_eval(genres) is a list of dictionaries.

#Thus by using this function, we later select the necessary genres and keywords corresponding to the particular movie which will help in increasing the similarity index between two movies.

new_df.loc[19995]['genres']
def genre(genres):
    g = [temp['name'] for temp in ast.literal_eval(genres)]
    return g
new_df['genres'] = new_df['genres'].apply(genre)
new_df['keywords'] = new_df['keywords'].apply(genre)
new_df.head()
#Similarly we perfom a similar operation on the cast that primarily includes the five major actors and the directors for any given film.

new_df['cast']=creds['cast']
new_df.iloc[0]['cast']
def get_actor(cast):
    l =[temp['name'] for i,temp in  enumerate(ast.literal_eval(cast),0) if i<5]
    r =[temp['character'] for i,temp in  enumerate(ast.literal_eval(cast),0) if i<5]
    return l+r
new_df['cast'] = new_df['cast'].apply(get_actor)
new_df.loc[268]['cast']
new_df['crew']=creds['crew']
new_df.iloc[0]['crew']
def director(crew):
    l = [temp['name'] for temp in ast.literal_eval(crew) if temp['job']=='Director']
    return l
new_df['director'] = new_df['crew'].apply(director)
new_df.iloc[0]['director']
def producer(crew):
    l = [temp['name'] for temp in ast.literal_eval(crew) if temp['job']=='Producer']
    return l
new_df['producer'] = new_df['crew'].apply(producer)
new_df.iloc[0]['producer']
#We transform the overview of the film such that all the stop words (words which bring no key imprtance to the sentence but are simply used for structure) are removed (Cell 45) and all the words deriving from the same family (for example - play, played, playing) are clubbed into a single one for checking similarity. We also ensure everything is in lowercase for homogeneous and more accurate results.

new_df['overview'] = new_df['overview'].str.split(' ')
new_df['overview'] = new_df['overview'].apply(lambda x:[temp.strip(",.") for temp in x])
new_df['overview']
new_df.drop('crew',axis=1,inplace=True)
new_df.head()
new_df.iloc[0]['overview']
#Now we club the full names of the cast as well as the genres to ensure two actors with same name (for example= Bruce Wayne and Bruce Lee) do not end up having an effect on similarity as these would mean more False Positive results and less precise results.

# a = new_df.iloc[0]['genres']
# a = [x.replace(" ","") for x in a]
new_df['genres'] = new_df['genres'].apply(lambda x:[temp.replace(" ","") for temp in x])
new_df['keywords'] = new_df['keywords'].apply(lambda x:[temp.replace(" ","") for temp in x])
new_df['director'] = new_df['director'].apply(lambda x:[temp.replace(" ","") for temp in x])
new_df['producer'] = new_df['producer'].apply(lambda x:[temp.replace(" ","") for temp in x])
new_df['cast'] = new_df['cast'].apply(lambda x:[temp.replace(" ","") for temp in x])
new_df.head()
new_df['tag'] = new_df['genres']  + new_df['keywords'] + new_df['title'].apply(lambda x:list(x))
new_df['tag2'] = new_df['cast'] + new_df['director'] + new_df['producer']
new_df['tag'] = new_df['tag'].apply(lambda x: " ".join(x))
new_df['tag2'] = new_df['tag2'].apply(lambda x: " ".join(x))
new_df['overview'] = new_df['overview'].apply(lambda x: " ".join(x))
new_df['tag2']
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def club(tag):
    l=[ps.stem(temp) for temp in tag.split()]
    str = " ".join(l)
    return str 
# club(" going gone go")
new_df['overview'] = new_df['overview'].apply(club)
# new_df['tag'][19995]
#The tags we created are used for calculating similarity scores based on different weights given to each of this. This technique of merging the words into a single string and finding similiarities is called "Box of Words". Here we create a 2D array of all the the top "x" max_features which have recurred multiple times, where each number represents how many times the word has been featured. Therefore, two movies having maximum common features are the ones which are most closely related and hence most similar.

#he CountVectoriser creates the vectors in space on which the cosine similarity is calculated.

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(lowercase=True,max_features=5100,stop_words='english')
cv2 = CountVectorizer(lowercase=True,max_features=5000, stop_words='english')
cv3 = CountVectorizer(lowercase=True,max_features=4000)
cv.fit(new_df['overview'])
count_vec = cv.transform(new_df['overview'])
count_vec_tag = cv2.fit_transform(new_df['tag'])
count_vec_tag_cast = cv3.fit_transform(new_df['tag2'])
count_vec = count_vec.toarray()
count_vec_tag = count_vec_tag.toarray()
count_vec_tag_cast = count_vec_tag_cast.toarray()
count_vec_tag_cast
# cv.get_feature_names_out()
np.set_printoptions(threshold=np.inf)
cv3.get_feature_names_out()
#Note that we are working in n-Dimentional space and hence we use cosing distance. The key advantage is that we do not need to scale the data as the magnitude does not matter but only the angle between the vectors. Thus cosine distance gives a better similarity index as compared to Eucledian distance.

from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
#Here we take weighted averages of all the three components used for calculating similarity. Since there are relatively less keywords and genres yet 40% weightage is given to it signifying higher importance of it in calculating similarity role.

sim = cosine_similarity(count_vec)*0.4
sim2 = cosine_similarity(count_vec_tag)*0.4
sim3 = cosine_similarity(count_vec_tag_cast)*0.2
simf = sim + sim2 + sim3
simf[0]
final_df = new_df
final_df = final_df.reset_index()
final_df.loc[0]
def movie_list(sim_score):
    l = sorted(list(enumerate(sim_score),key=lambda x:x[1],reverse=True))
    return l
def similar_movie(movie_name):
    ind = final_df[final_df['title'].apply(lambda x:x.lower())== movie_name.lower()].index[0]
    sim_score = simf[ind]
    movies = sorted(list(enumerate(sim_score)),reverse=True,key=lambda x:x[1])[1:20]
    for i in movies:
        print(new_df.iloc[i[0]].title)
similar_movie("Spider-Man 3")
#List of Movies to the given model
movie_list = final_df.sort_values(by=['vote_count'],ascending=False)['title']
movie_list.head(50)
 
 


# In[51]:


def movie_list(sim_score):
    l = sorted(list(enumerate(sim_score),key=lambda x:x[1],reverse=True))
    return l
dff=[]
def similar_movie(movie_name):
    ind = final_df[final_df['title'].apply(lambda x:x.lower())== movie_name.lower()].index[0]
    sim_score = simf[ind]
    movies = sorted(list(enumerate(sim_score)),reverse=True,key=lambda x:x[1])[1:20]
    for i in movies:
        print(new_df.iloc[i[0]].title)
    
        


# In[52]:


similar_movie('Ong Bak 2')
 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




