# NlpEmpatheticResponse
Providing an empathetic response to a question using sequence to sequence models

This project takes questions and responses given by mental health service providers to train a sequence to sequence model that returns an empathetic response to a question or statement provided by the user.

The initial part of the project is data cleaning and filtering of the points that can be useful for this model

These data points are then converted into a vocabulary to retrieve pretrained word vectors (Glove vectors are used in this case)

An encoder decoder model is fed with these vectors and trained to model the empathetic responses from the questions provided by the user.

### Steps to run the model:

1) Download the glove model from [here](https://archive.org/download/glove.6B.50d-300d/glove.6B.50d.txt)
2) Change all the paths provided in utils/constants.py to the paths in your local system - For example) Glove model path is glove_model_path
3) Run train_model.py in train\train_model.py
   *   *python train_model.py*
      
### Steps to Predict the Model:

1) Run the apps.py file - a web application interface will start 
 Replace {sentence} with a sentence to predict and run the following:
  *  *http://127.0.0.1:8080/predict_sent/{sentence}* 


### Challenges with this project:
1) Very less data to model
2) Embeddings for this model- which embeddings to use for training
3) The model parameters and architecture for this particular kind of model

### Future Work
1) Create more data and use it for training the model
2) Get semantic similar sentences to the training data to increase the training data points
2) Use BERT embeddings and train the model