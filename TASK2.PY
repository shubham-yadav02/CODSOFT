import pandas as pd
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy

# Step 1: Load the data
# Load the ratings and movies data
ratings = pd.read_csv('ratings.csv')  # Update with the correct path
movies = pd.read_csv('movies.csv')    # Update with the correct path

# Step 2: Prepare the data for Surprise library
reader = Reader(rating_scale=(1, 5))  # Ratings are between 1 and 5
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Step 3: Split the data into training and test sets (80% train, 20% test)
trainset, testset = train_test_split(data, test_size=0.2)

# Step 4: Train the model using Singular Value Decomposition (SVD)
model = SVD()
model.fit(trainset)

# Step 5: Make predictions on the test set
predictions = model.test(testset)

# Step 6: Evaluate the model
print(f"RMSE: {accuracy.rmse(predictions):.4f}")  # Root Mean Squared Error
print(f"MAE: {accuracy.mae(predictions):.4f}")    # Mean Absolute Error

# Step 7: Predict rating for a specific user and movie (example: user 1 and movie 50)
user_id = 1
movie_id = 50
predicted_rating = model.predict(user_id, movie_id).est
print(f"Predicted rating for user {user_id} and movie {movie_id}: {predicted_rating:.2f}")

# Optional: Recommend movies for a specific user
def recommend_movies(user_id, top_n=5):
    # Get a list of all movie IDs
    all_movie_ids = movies['movieId'].tolist()
    
    # Get the movies that the user has not rated yet
    rated_movie_ids = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
    
    # Predict ratings for the unrated movies
    predictions = [model.predict(user_id, movie_id) for movie_id in unrated_movie_ids]
    
    # Sort predictions by the estimated rating (highest first)
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get the top N recommended movies
    top_predictions = predictions[:top_n]
    
    recommended_movies = []
    for prediction in top_predictions:
        movie_id = prediction.iid
        movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
        recommended_movies.append((movie_title, prediction.est))
    
    return recommended_movies

# Example: Get the top 5 recommended movies for user 1
recommended_movies = recommend_movies(user_id=1, top_n=5)
print(f"\nTop 5 recommended movies for user {user_id}:")
for movie, rating in recommended_movies:
    print(f"{movie}: Predicted rating = {rating:.2f}")