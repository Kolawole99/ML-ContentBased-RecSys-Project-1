#===========================================IMPORTING=========================================
import pandas as pd
from math import sqrt
import numpy as np



#=========================================PREPROCESSING=====================================

#==================================Reading the data into the project=======================
#Storing the movie information into a pandas dataframe
movies_df = pd.read_csv('data/movies.csv')
#Storing the user information into a pandas dataframe
ratings_df = pd.read_csv('data/ratings.csv')
#Head is a function that gets the first N rows of a dataframe. N's default is 5.
movies = movies_df.head()
print(movies)

#============================================Movies============================================
#=============remove the year from the title column and store in a new year column========================
#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
print(movies_df.head())

#=============================Split the Genre column also into multiple columns============================
#Every genre is separated by a | so we simply have to call the split function on |
movies_df['genres'] = movies_df.genres.str.split('|')
print(movies_df.head())

#==============================Copying the movie dataframe into a new one==========================
moviesWithGenres_df = movies_df.copy()
#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
print(moviesWithGenres_df.head())

#===========================================Ratings=========================================
ratings_df.head()

#Drop timestamps from the dataframe
ratings_df = ratings_df.drop('timestamp', 1)
print(ratings_df.head())



#==============================CONTENT BASED RECOMMENDATION SYSTEM==============================

#===============================cCreate a user input to recommend movies to=======================
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)
print(inputMovies)

#==============================Add movie id to input user==============================
#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
print(inputMovies)

#======================Preformatting to learn the input's preferences================
#Filtering out the movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
print(userMovies)
#Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
#Dropping unnecessary issues due to save memory and to avoid issues
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
userGenreTable

#===================Learning the inputs preference==================================
# To do this, we're going to turn each genre into weights. We can do this by using the input's reviews and multiplying them into the input's genre table and then summing up the resulting table by column. This operation is actually a dot product between a matrix and a vector, so we can simply accomplish by calling Pandas's "dot" function.
inputMovies['rating']
#Dot produt to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
#The user profile
userProfile

#============================extracting the genre table from the original dataframe===============
#Now let's get the genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
#And drop the unnecessary information
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print(genreTable.head())
genreTable.shape

#=================================Recommending by the weighted average=============================
#Multiply the genres by the weights and then take the weighted average
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()
#Sort our recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
#Just a peek at the values
recommendationTable_df.head()
#The final recommendation table
print(movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())])



# Advantages and Disadvantages of Content-Based Filtering
    
#     Advantages
#     Learns user's preferences
#     Highly personalized for the user
    
#     Disadvantages
#     Doesn't take into account what others think of the item, so low quality item recommendations might happen
#     Extracting data is not always intuitive
#     Determining what characteristics of the item the user dislikes or likes is not always obvious