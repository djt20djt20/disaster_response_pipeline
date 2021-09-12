# Disaster Response Pipeline
This repo contains the code for my disaster-response machine learning pipeline. The purpose of this model is to be able to take incomming messages, and predict to which
rescue department the message ought to be sent.
### Libraries
In this project, I used the following libraries:
Flask<br/>
sqlalchemy<br/>
pandas <br/>
numpy<br/>
plotly<br/>
nltk<br/>
sklearn<br/>
joblib<br/>
scikit-learn==0.19.1<br/>
You can install these by typing 'pip install -r requirements.txt' into a command prompt.

### File description
The **data** folder contains the raw data, and a .py file for wrangling that data and saving it as an SQL database<br/>
The **models** folder contains the code for the model, which takes in the SQL data saved above<br/>
The **templates** folder contains templates for the associated webpage<br/>
**app.py** contains a flask app, along with three data visualizations


