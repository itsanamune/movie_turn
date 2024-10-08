from surprise import SVD, accuracy
from surprise.model_selection import train_test_split
import pickle

def train_and_evaluate_model(data):
    # Split the data into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

    # Train the SVD model
    model = SVD()
    model.fit(trainset)

    # Make predictions on the test set
    predictions = model.test(testset)

    # Calculate RMSE
    rmse = accuracy.rmse(predictions)

    return model, rmse

def save_model(model, filename='movie_recommender_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data

    data, _ = load_and_preprocess_data()
    model, rmse = train_and_evaluate_model(data)
    print(f"Model RMSE: {rmse}")
    save_model(model)
    print("Model saved successfully.")