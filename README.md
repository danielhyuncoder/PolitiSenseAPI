# PolitiSenseAPI
<br />
<h2>Start: </h2>
<p>Install the model from the google drive: <a href="https://drive.google.com/drive/folders/1OZ6ud4rPmYYtKNXFpiIa1-CNdI3vwRSU?usp=sharing">here</a></p>
<p>Run "training.py" first in order to fit a model based on the following dataset: <a href="https://www.kaggle.com/datasets/kapastor/democratvsrepublicantweets">Democrats Vs Republican Tweets</a> on Kaggle.com (Download "ExtractedTweets.csv").</p>
<h2>Features</h2>
<p>The API is built off of FastAPI and has one POST endpoint: "/predict/text" which takes in the body, {"text" : str}. Through the "text" variable in the body, it will predict the overall political sentiment of it and will return the string version of the prediction. Since the output is based on a sigmoid activation function, it will give you a probability between the range of 0 - 1, with < 0.5 meaning Democratic and 0.5 > meaning Republican</p>
  <br />
<h2>Model:</h2>
<p>The model is built on the Tensorflow and Keras frameworks, with the text preprocessing and neural network completely based on it. In terms of the model architecture, it is composed of an Embedding layer (Of which has a input dim of the size of the vocab), an LSTM layer (With 32 units and an activation of ReLU), a normal Dense layer (With 10 units and an activation of ReLU, and finally a Dense layer (with 1 unit, and an activation of sigmoid (Which is what gives the probability distribution between 0-1).</p>
