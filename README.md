# Seq2Seq Learning using Pytorch and implementing a REST API to predict the model's results

- The project's inputs are questions and answers given by mental health service providers to train a sequence to sequence model and predict the answer given by the model (Data is available [here](https://github.com/nbertagnolli/counsel-chat/tree/master/data))
- The word vectors for the inputs are taken from glove embeddings (transfer learninng of pretrained model)
- The model predictions are bad due to the limited number of training data points
- This project shows how to structure a REST API for a trained pytorch model and use it to predict outputs at realtime.

### Workflow of the model training
* The initial part of the project is data cleaning and filtering of the points that can be useful for this model

* These data points are then converted into a vocabulary to retrieve pretrained word vectors (Glove vectors are used in this case)

* An encoder decoder model is fed with these vectors and trained to model the responses of the questions provided by the user

* Implemented a Bidirectional GRU model at encoder and a GRU model with attention at decoder

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


### Future Work
1) Create more data and use it for training the model
2) Get semantic similar sentences to the training data to increase the training data points
3) Use BERT embeddings and train the model
4) Expand vocabulary of the model since it uses only the words in training as the vocab. (Bert Model can help with this)

