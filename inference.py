import pickle
import json
import numpy as np

def model_fn(model_dir):
    with open(f"{model_dir}/movie_recommender_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    user_id = input_data['user_id']
    movie_ids = input_data['movie_ids']
    
    predictions = [model.predict(user_id, movie_id).est for movie_id in movie_ids]
    return predictions

def output_fn(predictions, response_content_type):
    if response_content_type == 'application/json':
        return json.dumps(predictions)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")