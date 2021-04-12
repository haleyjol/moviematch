# MovieMatch
Using datasets from MovieLens and IMDB with machine learning models (SVD &amp; KNNBaseline) to create a recommender system that recommends for two users.



File Description
==========================================================================================

Notebook 1: Includes the exploratory data analysis process and pre-processing steps
used to create the feature-utility matrix.

Notebook 2: Includes the short EDA for the ratings data, the model selection and 
hyperparameter optimization for both models used as well as some initial model evaluation.

Notebook 3: Includes calculation of precision and recall, functions to hybridize the 
ratings of both models, functions to return top films for multiple users, functions 
to return popular films for specified genres and functions to create datasets for new
users ratings.

Notebook 4: A demo notebook meant to be used for demonstrating the functionality of the
system and tweaking parameters without fear of damaging parts of the original system. 

Final Report: A short summary of the project.

Demo Video: A short video demo-ing the use of the functions in Notebook 4. Useful if you
do not have the proper application to use .ipynb files.


DATASETS
==========================================================================================

Due to the size of the datasets, all initial imports were done in Google Colab, so the 
files will not included in this folder. The links to the files will be added in below.

MovieLens Latest (27M): used for link-key data
https://files.grouplens.org/datasets/movielens/ml-latest.zip

MovieLens 20M: used for movies & tags data
https://files.grouplens.org/datasets/movielens/ml-20m.zip

MovieLens 1M: used for ratings data
https://files.grouplens.org/datasets/movielens/ml-1m.zip


USED FOR AUXILIARY INFORMATION

IMDB Title Basics: 
https://datasets.imdbws.com/title.basics.tsv.gz

IMDB Title Ratings:
https://datasets.imdbws.com/title.ratings.tsv.gz

IMDB Title Crew:
https://datasets.imdbws.com/title.crew.tsv.gz

IMDB Name Basics:
https://datasets.imdbws.com/name.basics.tsv.gz
