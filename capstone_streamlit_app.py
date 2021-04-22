# import relevant packages
import streamlit as st
# setting view to wide
st.set_page_config(layout="wide")

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

# define function for loading in data
@st.cache
def load_data(path):
    df = pd.read_csv(path)
    return df

# define function for loading in numpy array
@st.cache
def load_array(path):
    array = np.load(path)
    return array

# load in reference dataframe & similarities array from local drive
movies_reference = load_data('My Datasets/movies_reference.csv')
movie_similarities = load_array('My Datasets/movie_similarities.npy')

# define function for loading in reference matrix
@st.cache
def load_reference_matrix(path, ref_df):
    # reading in movie utility matrix 
    df = pd.read_csv(path, index_col=0)
    # taking out the first 20 columns, these are all of the genres
    df = df.iloc[:,:20]
    # creation of new reference df, with only the movieId & genres 
    df = pd.merge(ref_df, df, on='movieId', how='left')
    return df

# loading in the reference data
genre_reference_df = load_reference_matrix('My Datasets/movies_utility.csv', movies_reference)

# defining function for loading in model predictions
@st.cache(allow_output_mutation=True)
def load_model(path):
    model_path = open(path, 'rb')
    predictions = pickle.load(model_path)
    return predictions

# loading in predictions
predictions = load_model('Models/mixed_predictions.pkl')


# defining function to get the top N movies for all users
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

# calling function to instantiate the top N dict for each user
top_dict = get_top_N_movies(predictions, n=100, threshold=3.5)




# defining function for getting the matched ratings for two users
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
        # if the movie does not exist in the reference dataframe, then do not include
        except TypeError:
            pass
    
    # threshold for low popularity films 
    top_for_both = top_for_both[top_for_both['IMDB Vote Count'] > 45000]
    top_for_both = top_for_both[top_for_both['Year Of Release'] > 1965]
    
    top_for_both.reset_index(drop=True, inplace=True)
    
    # return the dataframe
    return top_for_both



# defining function to create the top 5 similarities dataframe
def similar_items_to_top_5(user1, user2):
    ''' Takes in two user ids, returns a dataframe of top 5 similar movies 
    to the top 5 movies in the two users matched dataframe'''
    
    # calling previous function to get the top 5 films
    df = ratings_for_two(user1, user2)
    
    # instantiating list of dataframes to be concatenated
    to_concat_list = []
    
    # for the first five films
    for i, iid in enumerate(df['MovieId'].iloc[:5]):
        
        # use the iid to search for the index location of the film
        movie_index = (movies_reference[movies_reference['movieId'] == iid].index)
        # the original film title from the top films
        similar_to = str(movies_reference['title'][movies_reference['movieId'] == iid].values).strip("[\W']")
        
        # create dataframe with the top 5 similar films, from the similarites array
        i = pd.DataFrame({'Movie':movies_reference['title'],
                          'Year of Release':movies_reference['year_of_release'],
                          'IMDB Rating':movies_reference['averageRating'],
                          'Similarity Score': np.array(movie_similarities[movie_index, :].squeeze()),
                         'Similar To:': similar_to})
        
        # sort the dataframe by the similarity score, take only top 5 similar
        # not including the first, bc that will be the original film
        i = i.sort_values(by='Similarity Score', ascending=False).iloc[1:6,:]
        
        # add the id of the dataframe to the list
        to_concat_list.append(i)
    
    # concatenate all dataframes together
    similar_df = pd.concat(to_concat_list, axis=0)
    
    # set the index of the dataframe to the movie's from the original list
    similar_df = similar_df.set_index(['Similar To:'])
    
    return similar_df



# defining a function to get the top in each genre
@st.cache
def top_in_genre(genre_choice):
    ''' Takes in user input of genre, returns the top 15 films in each genre
    by their overall rating on IMDB and the number of votes.'''
    
    best_in_genre = genre_reference_df[['title', 'year_of_release','runtimeMinutes', 'numVotes', 'averageRating']][genre_reference_df[genre_choice] == 1].sort_values(['numVotes', 'averageRating'], ascending=False).head(15)
    
    return best_in_genre


# Getting top ratings for two
'''
Enter IDs below to get your top movie matches!
'''
# using user input for the ids
id1 = st.number_input(label='ID 1', value=1)
id2 = st.number_input(label='ID 2', value=1)

'''
Top movie matches
'''
# call function with user input ids
st.write(ratings_for_two(id1, id2))

# pulls up the top 5 similar films automatically with input ids
'''
Similar films to the top 5
'''
st.write(similar_items_to_top_5(id1, id2))


# Search for popular films by genre
st.subheader('Top movies by genre')
# picking the genre using a dropdown menu 
genre_pick = st.selectbox('Select a Genre', genre_reference_df.drop(columns=['movieId', 'title', 'year_of_release', 'averageRating', 'numVotes', 'runtimeMinutes']).columns).capitalize()
# calling the function with the genre pick from the drop down menu
st.write(top_in_genre(genre_pick))


# Search database for favourite films
st.subheader('Search our database for your favourite films!')

# defining function to make up the similarities dataframe
@st.cache
def make_similar_df(movie_index):
    # creation of dataframe
    similarity_df = pd.DataFrame({'Movie':movies_reference['title'],
        'Year of Release':movies_reference['year_of_release'],
        'Similarity Score': np.array(movie_similarities[movie_index, :].squeeze())})
    # sort dataframe by similarity and only show top 15
    similarity_df.sort_values(by='Similarity Score', ascending=False)

    return similarity_df

# using user input to search the database by keyword
search_text = st.text_input('Search:')
st.dataframe(movies_reference[['movieId', 'title', 'year_of_release']][movies_reference['title'].str.contains(search_text)])

'''
If you would like to see similar films, enter movie id below:
'''
# using user input of movie Id to create similarities dataframe with 
movieid = int(st.number_input('Movie Id', value=1, min_value=1))
# using movie Id to get index location reference dataframe
index_loc_in_reference = (movies_reference[movies_reference['movieId'] == movieid].index)
# calling function with the index location in the similarities array
st.write(make_similar_df(index_loc_in_reference))
