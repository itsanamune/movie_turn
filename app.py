from flask import Flask, render_template, request, jsonify
import boto3
import json

app = Flask(__name__)

# Replace with your actual endpoint name
SAGEMAKER_ENDPOINT = 'your-sagemaker-endpoint-name'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    movie_ids = [int(id) for id in request.form['movie_ids'].split(',')]
    
    # Call SageMaker endpoint
    client = boto3.client('sagemaker-runtime')
    response = client.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT,
        ContentType='application/json',
        Body=json.dumps({'user_id': user_id, 'movie_ids': movie_ids})
    )
    
    predictions = json.loads(response['Body'].read().decode())
    
    # Sort movies by predicted rating
    sorted_movies = sorted(zip(movie_ids, predictions), key=lambda x: x[1], reverse=True)
    
    return jsonify(sorted_movies)

if __name__ == '__main__':
    app.run(debug=True)