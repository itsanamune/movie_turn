import pandas as pd
import numpy as np
from surprise import Dataset, Reader

def load_and_preprocess_data(ratings_file='ml-latest-small/ratings.csv', movies_file='ml-latest-small/movies.csv'):
    # Load ratings and movies data
    ratings = pd.read_csv(ratings_file)
    movies = pd.read_csv(movies_file)

    # Merge ratings and movies data
    df = pd.merge(ratings, movies, on='movieId')

    # Filter out movies with less than 10 ratings
    movie_counts = df['movieId'].value_counts()
    df = df[df['movieId'].isin(movie_counts[movie_counts >= 10].index)]

    # Create a Surprise dataset
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

    return data, df

if __name__ == "__main__":
    data, df = load_and_preprocess_data()
    print("Data loaded and preprocessed successfully.")
    print(f"Number of ratings: {len(df)}")
    print(f"Number of unique users: {df['userId'].nunique()}")
    print(f"Number of unique movies: {df['movieId'].nunique()}")