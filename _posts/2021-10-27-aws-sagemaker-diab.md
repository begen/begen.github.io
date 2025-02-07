---
layout: post
title: "Amazon SageMaker- Predict Whether a Customer Will Enroll for a Certificate of Deposit"
subtitle: "In this tutorial, we build, train and deploy a machine learning model with AWS SageMaker"
date: 2020-10-01 23:45:13 -0400
background: '/img/bg-post.jpg'
---

<p>In this tutorial, we will be working with Amazon SageMaker to build, train and deploy a machine learning model using XGBoost algorithm.</p>

<p>We will be working with a bank dataset and we will try to predict whether a customer will enroll for a certificate of deposit or CD.</p>

<p>In order to work, we need to have AWS account.</p>

<p>We will follow following steps: 1) Create a notebook 2) Prepare our data 3) Train the model to learn from the training dataset 4) Deploy the model 5) Evaluate performance</p>

<p>Once you login to AWS SageMaker, create instance of a notebook, IAM role and select S3 bucket.</p>

<p>In Jupyter, create a new conda_python3 notebook.</p>

<p>Below code imports the required libraries amd defines environmental variables needed to prepare the data, train the ML Model, and deploy ML Model

</p>

#### Prepare the data

    # import libraries
    import boto3, re, sys, math, json, os, sagemaker, urllib.request
    from sagemaker import get_execution_role
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import Image
    from IPython.display import display
    from time import gmtime, strftime
    from sagemaker.predictor import csv_serializer

    # Define IAM role
    role = get_execution_role()
    prefix = 'sagemaker/DEMO-xgboost-dm'
    my_region = boto3.session.Session().region_name # set the region of the instance

    # this line automatically looks for the XGBoost image URI and builds an XGBoost container.
    xgboost_container = sagemaker.image_uris.retrieve("xgboost", my_region, "latest")

    print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + xgboost_container + " container for your SageMaker endpoint.")
    
    
  <p>We need storage where we will be upload and store our data. For that, we will be using Amazon resources and we will to create S3 bucket in one of the AWS regions. Below code creates a bucket with a unique name that will be reserved for us until it is deleted. We choose a region that is close to optimize latency and to minimize cost. 
    </p>
    
    bucket_name = 'your-s3-bucket-name' # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
    s3 = boto3.resource('s3')
    try:
    if  my_region == 'us-east-1':
      s3.create_bucket(Bucket=bucket_name)
    else: 
      s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
    print('S3 bucket created successfully')
    except Exception as e:
    print('S3 error: ',e)

    
  <p>In below code, we copy the file denoted by the URL to a local S3 bucket. In our case, the file is csv and  load it to datafame and assign to a variable model_data. 
  </p>
  
    try:
      urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
    print('Success: downloaded bank_clean.csv.')
    except Exception as e:
    print('Data load error: ',e)

    try:
    model_data = pd.read_csv('./bank_clean.csv',index_col=0)
    print('Success: Data loaded into dataframe.')
    except Exception as e:

<p>
    Next, we shuffle date and split data into train and test data with 70%  and 30% ratio, respectively. 
</p>
    
    train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
    print(train_data.shape, test_data.shape)


  #### Training the ML Model:
  


<p>
Our data contains 0 and 1 meaning that customers did not sign up for certificate of deposit and 1 meaning customers signed up.  

Following code concatenates those two columns and saves as one column to train.csv file and removes indexes and headers. 
</p>


    pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
    boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
    s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')

<p>
Next step would be to setup Amazon SageMaker, create instance for XGBoost model and we need to define the hyperparameters.  
</p>

    sess = sagemaker.Session()
    xgb = sagemaker.estimator.Estimator(xgboost_container,role, instance_count=1, instance_type='ml.m4.xlarge',output_path='s3://{}/{}/output'.format(bucket_name, prefix),sagemaker_session=sess)
    xgb.set_hyperparameters(max_depth=5,eta=0.2,gamma=4,min_child_weight=6,subsample=0.8,silent=0,objective='binary:logistic',num_round=100)


<p>
Now, we will train the model using training data 
</p>

    xgb.fit({'train': s3_input_train})


#### Deploy the model
<p>
Once we train the model, we will deploy the trained model to an endpoint and run the model to create predictions. 
</p>

    xgb_predictor = xgb.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')


<p>
Now, we will be using our test data to evaluate our model
</p>

    from sagemaker.serializers import CSVSerializer

    test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array
    xgb_predictor.serializer = CSVSerializer() # set the serializer type
    predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
    predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
    print(predictions_array.shape)


#### Evaluation of Model and Performance

    cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
    tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
    print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
    print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
    print("Observed")
    print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
    print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))





