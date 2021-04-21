# import relevant packages
import streamlit as st

import pandas as pd
import numpy as np

from surprise import dump
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
import pickle


#webpage title
st.title("Movie Match")

'''
A recommendation system for couples!

'''

# loading in data
@st.cache
def load_data(path):
    df = pd.read_csv(path)
    return df

@st.cache
def load_array(path):
    array = np.load(path)
    return array

movies_reference = load_data('My Datasets/movies_reference.csv')
movie_similarities = load_array('My Datasets/movie_similarities.npy')

# loading in reference matrix
@st.cache
def load_reference_matrix(path, ref_df):
    # reading in movie utility matrix 
    df = pd.read_csv(path, index_col=0)
    # taking out the first 20 columns, these are all of the genres
    df = df.iloc[:,:20]
    # creation of new reference df, with the movieId & genres 
    df = pd.merge(ref_df, df, on='movieId', how='left')
    return df

genre_reference_df = load_reference_matrix('My Datasets/movies_utility.csv', movies_reference)

# loading in models
@st.cache(allow_output_mutation=True)
def load_model(path):
    model_path = open(path, 'rb')
    predictions = pickle.load(model_path)
    return predictions

predictions = load_model('Models/mixed_predictions.pkl')


# Define function to load a model we've previously built
@st.cache(allow_output_mutation=True)
def load_model(path):
    model = joblib.load(path)
    return model



# FUNCTION TO GET TOP RATINGS FOR EACH USER
def get_top_N_movies(predictions, n=10, threshold=3.5):
    ''' Takes in dictionary of user ids and associated true rating/predited rating tuples.
    Returns the top N per user, as specified in the function call.
    Required format: dictionary of lists of tuples, ex. uid: [(iid, est)]
    Where uid = user id, iid = item id and est = predicted rating of item by user.
    Arguments: predictions, top N ratings, ratings threshold'''
    
    user_top_n_films = defaultdict(list)
    
    # going through the list of tuples, sorting by the predicted rating
    # slicing out only the top n for each user
    for uid, ratings in predictions.items():
        
        # sort the tuples in the keys by the predicted rating
        # lambda function calls x as each tuple, indexes to the first item (est)
        # sorts the tuple list by that estimated rating
        ratings.sort(key=lambda x: x[1], reverse=True)
        
        # separates off the top n ratings for each user
        user_top_n_films[uid] = ratings[:n]
    
    # deleting data in the top n ratings that is not within our rating threshold
    for uid, ratings in user_top_n_films.items():
        # if the estimated rating is below the threshold
        for i, rating in enumerate(ratings):
            if rating[1] < threshold:
                ratings.pop(i)
        
    return user_top_n_films

top_dict = get_top_N_movies(predictions, n=100, threshold=3.5)

# FUNCTION TO GET RATINGS FOR 2
def ratings_for_two(user1, user2):
    '''Takes in two user ids, returns a dataframe of movies that would be recommended for both'''
    
    # uses top_n function to create list of movies that each user would like
    movies_for_one = top_dict[user1]
    movies_for_two = top_dict[user2]
    
    # instatiate empty list for the combined movies
    movies_for_both = []
    
    # filling movies for both list, averaging the couple rating
    for (iid1, rating1) in movies_for_one:
        for (iid2, rating2) in movies_for_two:
            # IF the movie ids match, append the id and an averaged rating to the list
            if iid1 == iid2:
                movies_for_both.append((iid1, ((rating1+rating2)/2)))
            
    
    # sort the list by the averaged rating
    movies_for_both.sort(key=lambda x: x[1], reverse=True)
    
    # instantiating dataframe for visibility
    top_for_both = pd.DataFrame(columns=['Title', 'Year Of Release', 'IMDB Rating /10', 'Couple Pred Rating /5', 'IMDB Vote Count', 'MovieId'])
    
    # for each movie + rating in the movies for both list
    for i, (iid, rating) in enumerate(movies_for_both):
        try:
        # populate dataframe, using the movies reference table for the values
            top_for_both.loc[i] = [str(movies_reference['title'][movies_reference['movieId'] == iid].values).strip("(?:[''])"),
                                   int(movies_reference['year_of_release'][movies_reference['movieId'] == iid].values),
                                   float(movies_reference['averageRating'][movies_reference['movieId'] == iid].values),
                                   round(rating,1),
                                   int(movies_reference['numVotes'][movies_reference['movieId'] == iid].values),
                                   int(movies_reference['movieId'][movies_reference['movieId'] == iid].values)]
        except TypeError:
            pass
    
    top_for_both = top_for_both[top_for_both['IMDB Vote Count'] > 45000]
    top_for_both = top_for_both[top_for_both['Year Of Release'] > 1965]
    
    top_for_both.reset_index(drop=True, inplace=True)
    
    # return the dataframe
    return top_for_both



# FINIDNG SIMILAR TO TOP N FOR 2
def similar_items_to_top_5(user1, user2):
    ''' Takes in dataframe, returns a dataframe of top 5 similar movies 
    to the top 5 movies in the dataframe'''
    
    df = ratings_for_two(user1, user2)
    
    to_concat_list = []

    for i, iid in enumerate(df['MovieId'].iloc[:5]):

        movie_index = (movies_reference[movies_reference['movieId'] == iid].index)
        similar_to = str(movies_reference['title'][movies_reference['movieId'] == iid].values).strip("[\W']")

        i = pd.DataFrame({'Movie':movies_reference['title'],
                          'Year of Release':movies_reference['year_of_release'],
                          'IMDB Rating':movies_reference['averageRating'],
                          'Similarity Score': np.array(movie_similarities[movie_index, :].squeeze()),
                         'Similar To:': similar_to})

        i = i.sort_values(by='Similarity Score', ascending=False).iloc[1:6,:]

        to_concat_list.append(i)

    similar_df = pd.concat(to_concat_list, axis=0)
    
    similar_df = similar_df.set_index(['Similar To:', 'Movie'])
    
    return similar_df



# TOP IN EACH GENRE
@st.cache
def top_in_genre(genre_choice):
    ''' Takes in user input of genre, returns the top 15 films in each genre
    by their overall rating on IMDB and the number of votes.'''
    
    best_in_genre = genre_reference_df[['title', 'year_of_release','runtimeMinutes', 'numVotes', 'averageRating']][genre_reference_df[genre_choice] == 1].sort_values(['numVotes', 'averageRating'], ascending=False).head(15)
    
    return best_in_genre


# GET TOP RATED FILMS FOR 2
'''
Enter IDs below to get your top movie matches!
'''
id1 = st.number_input(label='ID 1', value=1)
id2 = st.number_input(label='ID 2', value=1)
st.write(ratings_for_two(id1, id2))
# PULLS UP TOP 5 SIMILAR AUTOMATICALLY
'''
Similar films to the top 5
'''
st.write(similar_items_to_top_5(id1, id2))


# SEARCH FOR POPULAR FILMS BY GENRE
st.subheader('Popular Films')
genre_pick = st.selectbox('Select a Genre', genre_reference_df.drop(columns=['movieId', 'title', 'year_of_release', 'averageRating', 'numVotes', 'runtimeMinutes']).columns).capitalize()
st.write(top_in_genre(genre_pick))


# SEARCH FOR FAVOURITE FILMS
st.subheader('Search for your Favourite Films!')

@st.cache
def make_similar_df(movie_index):
    # creation of dataframe
    similarity_df = pd.DataFrame({'Movie':movies_reference['title'],
        'Year of Release':movies_reference['year_of_release'],
        'Similarity Score': np.array(movie_similarities[movie_index, :].squeeze())})
    # sort dataframe by similarity and only show top 15
    similarity_df.sort_values(by='Similarity Score', ascending=False).head(15)

    return similarity_df

search_text = st.text_input('Search:')
st.dataframe(movies_reference[['movieId', 'title']][movies_reference['title'].str.contains(search_text)])

'''
If you would like to see similar films, enter movie id below:
'''
movieid = int(st.number_input('Movie Id', value=1, min_value=1))

index_loc_in_reference = (movies_reference[movies_reference['movieId'] == movieid].index)

st.write(make_similar_df(index_loc_in_reference))
