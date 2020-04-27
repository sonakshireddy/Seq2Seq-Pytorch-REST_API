# NlpEmpatheticResponse
Providing an empathetic response to a question using sequence to sequence models

This project takes questions and responses given by mental health service providers to train a sequence to sequence model that returns an empathetic response to a question or statement provided by the user.

The initial part of the project is data cleaning and filtering of the points that can be useful for this model

These data points are then converted into a vocabulary to retrieve pretrained word vectors (Glove vectors are used in this case)

An encoder decoder model is fed with these vectors and trained to model the empathetic responses from the questions provided by the user.

Currently, implemented a Bidirectional GRU model at encoder and a GRU model with attention at decoder

### Steps to run the model:

1) Download the glove model from [here](https://archive.org/download/glove.6B.50d-300d/glove.6B.50d.txt)
2) Change all the paths provided in utils/constants.py to the paths in your local system - For example) glove_model_path variable in the file needs to be changed to the path of the downloaded file above
3) Run save_glove_vectors.py in glove_emb/
  *    *python save_glove_vectors.py*
4) Run train_model.py in train/
   *   *python train_model.py*
      
### Steps to Predict the Model:

1) Run the apps.py file - a web application interface will start
2) Replace {sentence} with a sentence to predict and run the following:
  *  *http://127.0.0.1:8080/predict_sent/{sentence}* 


### Challenges with this project:
1) Very less data to model
2) Embeddings for this model- which embeddings to use for training
3) The model parameters and architecture for this particular kind of model
4) Model is not able to learn long sentences- model architecture needs to be changed

### Future Work
1) Create more data and use it for training the model
2) Get semantic similar sentences to the training data to increase the training data points
3) Use BERT embeddings and train the model
4) Expand vocabulary of the model since it uses only the words in training as the vocab. (Bert Model can help with this)

