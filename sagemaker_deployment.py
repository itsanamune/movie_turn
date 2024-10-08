import boto3
import sagemaker
from sagemaker.sklearn import SKLearnModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

def deploy_model_to_sagemaker(model_data_path, role_arn, instance_type='ml.t2.medium'):
    sagemaker_session = sagemaker.Session()
    
    sklearn_model = SKLearnModel(
        model_data=model_data_path,
        role=role_arn,
        entry_point='inference.py',
        framework_version='0.23-1',
        py_version='py3',
        instance_type=instance_type,
        sagemaker_session=sagemaker_session
    )
    
    predictor = sklearn_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )
    
    return predictor.endpoint_name

if __name__ == "__main__":
    # Replace these with your actual values
    model_data_path = 's3://your-bucket/movie_recommender_model.tar.gz'
    role_arn = 'arn:aws:iam::your-account-id:role/SageMakerRole'
    
    endpoint_name = deploy_model_to_sagemaker(model_data_path, role_arn)
    print(f"Model deployed successfully. Endpoint name: {endpoint_name}")