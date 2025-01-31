# #!/usr/bin/env python
# import requests
# import os
# import pandas as pd

# from sagemaker.analytics import TrainingJobAnalytics
# import sagemaker
# from sagemaker.estimator import Estimator
# import boto3
# import s3fs

# session = sagemaker.Session(boto3.session.Session())

# BUCKET_NAME = os.environ['BUCKET_NAME']
# PREFIX = os.environ['PREFIX']
# REGION = os.environ['AWS_DEFAULT_REGION']
# # Replace with your IAM role arn that has enough access (e.g. SageMakerFullAccess)
# IAM_ROLE_NAME = os.environ['IAM_ROLE_NAME']
# GITHUB_SHA = os.environ['GITHUB_SHA']
# ACCOUNT_ID = session.boto_session.client(
#     'sts').get_caller_identity()['Account']
# # Replace with your desired training instance
# training_instance = 'ml.m5.large'

# # Replace with your data s3 path
# training_data_s3_uri = 's3://{}/{}/boston-housing-training.csv'.format(
#     BUCKET_NAME, PREFIX)
# validation_data_s3_uri = 's3://{}/{}/boston-housing-validation.csv'.format(
#     BUCKET_NAME, PREFIX)


# output_folder_s3_uri = 's3://{}/{}/output/'.format(BUCKET_NAME, PREFIX)
# source_folder = 's3://{}/{}/source-folders'.format(BUCKET_NAME, PREFIX)
# base_job_name = 'boston-housing-model'


# # Define estimator object
# boston_estimator = Estimator(
#     image_uri=f'{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/my-app:latest',
#     role=IAM_ROLE_NAME ,
#     instance_count=1,
#     instance_type=training_instance,
#     output_path=output_folder_s3_uri,
#     code_location=source_folder,
#     base_job_name='boston-housing-model',
#     hyperparameters={'nestimators': 70},
#     environment={
#              "BUCKET_NAME": BUCKET_NAME,
#              "PREFIX": PREFIX,
#              "GITHUB_SHA": GITHUB_SHA,
#              "REGION": REGION,},

#     tags=[{"Key": "email",
#            "Value": "haythemaws@gmail.com"}])

# # training job triggered here using the specified Docker image, which contains the training-script.py
# boston_estimator.fit({'training': training_data_s3_uri,
#                       'validation': validation_data_s3_uri}, wait=False)


# training_job_name = boston_estimator.latest_training_job.name
# hyperparameters_dictionary = boston_estimator.hyperparameters()


# report = pd.read_csv(f's3://{BUCKET_NAME}/{PREFIX}/reports.csv')
# while(len(report[report['commit_hash']==GITHUB_SHA]) == 0):
#     report = pd.read_csv(f's3://{BUCKET_NAME}/{PREFIX}/reports.csv')

# res = report[report['commit_hash']==GITHUB_SHA]
# metrics_dataframe = res[['Train_MSE', 'Validation_MSE']]

# message = (f"## Training Job Submission Report\n\n"
#            f"Training Job name: '{training_job_name}'\n\n"
#             "Model Artifacts Location:\n\n"
#            f"'s3://{BUCKET_NAME}/{PREFIX}/output/{training_job_name}/output/model.tar.gz'\n\n"
#            f"Model hyperparameters: {hyperparameters_dictionary}\n\n"
#             "See the Logs in a few minute at: "
#            f"[CloudWatch](https://{REGION}.console.aws.amazon.com/cloudwatch/home?region={REGION}#logStream:group=/aws/sagemaker/TrainingJobs;prefix={training_job_name})\n\n"
#             "If you merge this pull request the resulting endpoint will be avaible this URL:\n\n"
#            f"'https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{training_job_name}/invocations'\n\n"
#            f"## Training Job Performance Report\n\n"
#            f"{metrics_dataframe.to_markdown(index=False)}\n\n"
#           )
# print(message)

# # Write metrics to file
# with open('details.txt', 'w') as outfile:
#     outfile.write(message)

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator

# Set up SageMaker session and role
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Define the S3 bucket and prefix
bucket = 'sample-sagemaker-cicd-tuto1'
prefix = 'bouston-housing-regression'

# Define the training data location
train_data = f's3://{bucket}/{prefix}/train.csv'
validation_data = f's3://{bucket}/{prefix}/validation.csv'

# Define the output path
output_path = f's3://{bucket}/{prefix}/output'

# Define the XGBoost container
container = sagemaker.image_uris.retrieve("xgboost", sagemaker_session.boto_region_name, "1.2-1")

# Create the Estimator
xgboost_estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=output_path,
    sagemaker_session=sagemaker_session
)

# Set hyperparameters
xgboost_estimator.set_hyperparameters(
    objective='reg:squarederror',
    num_round=100
)

# Define the data channels
train_input = TrainingInput(train_data, content_type='csv')
validation_input = TrainingInput(validation_data, content_type='csv')

# Start the training job
xgboost_estimator.fit({'train': train_input, 'validation': validation_input})

# Save the training job name
training_job_name = xgboost_estimator.latest_training_job.name
print(f'Training job name: {training_job_name}')

# Write metrics to file
with open('details.txt', 'w') as outfile:
    outfile.write(f'Training job name: {training_job_name}\n')
