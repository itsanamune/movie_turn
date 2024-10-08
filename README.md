# Movie Recommendation System

This project implements a movie recommendation system using collaborative filtering with Singular Value Decomposition (SVD). The system is trained on the MovieLens dataset and deployed on AWS SageMaker for real-time recommendations.

## How It Works

1. Data Preprocessing: The MovieLens dataset is loaded and preprocessed to handle missing values and filter low-rated movies.
2. Model Training: An SVD-based collaborative filtering model is trained using the Surprise library.
3. Model Evaluation: The model's performance is evaluated using Root Mean Squared Error (RMSE).
4. Deployment: The trained model is deployed to AWS SageMaker for real-time recommendations.
5. Web Interface: A Flask web app allows users to input a movie and see recommendations.

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/movie-recommender.git
   cd movie-recommender
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the MovieLens dataset and place it in the project directory.

4. Train the model:
   ```
   python model.py
   ```

5. Deploy the model to AWS SageMaker:
   ```
   python sagemaker_deployment.py
   ```

6. Run the Flask web app:
   ```
   python app.py
   ```

## Using the SageMaker Endpoint

To use the SageMaker endpoint for recommendations:

1. Send a POST request to the endpoint URL with the following JSON payload:
   ```json
   {
     "user_id": 1,
     "movie_ids": [1, 2, 3, 4, 5]
   }
   ```

2. The endpoint will return a list of predicted ratings for the given movie IDs.

## Running Locally

1. Ensure you have completed the Setup and Installation steps.
2. Run the Flask app: `python app.py`
3. Open a web browser and navigate to `http://localhost:5000`
4. Enter a user ID and a list of movie IDs to get recommendations.

## Deployment on AWS

1. Set up your AWS credentials and configure the AWS CLI.
2. Update the `sagemaker_deployment.py` file with your S3 bucket and IAM role ARN.
3. Run the GitHub Actions workflow to deploy the model to SageMaker and the web app to Elastic Beanstalk.

## CI/CD with GitHub Actions

The project includes a GitHub Actions workflow that automates the following steps:
1. Install dependencies
2. Train the model
3. Deploy the model to SageMaker
4. Deploy the web app to Elastic Beanstalk

To use this workflow, make sure to set up the following secrets in your GitHub repository:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.