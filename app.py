import json

import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib

from sqlalchemy import create_engine
import sklearn
from sklearn.ensemble import RandomForestClassifier
import nltk

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor():
    # this is a custom transformer, used in the pipeline. It's purpose is to work out whether the first word is a ver 
    
    def starting_verb(self, text):
        
        # get sentence tokenized text
        sentence_list = nltk.sent_tokenize(text)
        
        #loop through sentences
        for sentence in sentence_list:
            # tokenize the sentence
            sentence = tokenize(sentence)

            # this tags the words with a 'part of speach'
            pos_tags = nltk.pos_tag(sentence)

            
            #extract the first words and first tags
            if pos_tags == []:
                return False
            else:
                first_word, first_tag = pos_tags[0]
            
            # if the first tag is a verb, return true
            if first_tag in ['VB', 'VBP']:
                return True
        # if not, return false
        return False
    
    # does nothing
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # takes in a list of strings. Turns it into a series, and applies the 
        # 'starting verb' method to each 
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

# load data
engine = create_engine('sqlite:///data.db')
df = pd.read_sql_table('data', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    

    targets = list(set(df.columns).difference(set(['id', 'message', 'original', 'genre'])))
    target_counts = [df[output].sum() for output in targets]

    sv = StartingVerbExtractor()
    starting_verbs = df['message'].apply(lambda row : sv.starting_verb(row)).sum()
    starting_non_verbs = len(df) - starting_verbs

    starting_names = ['verb', 'non-verb']
    starting_counts = [starting_verbs,starting_non_verbs]
    # create visuals

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=targets,
                    y=target_counts
                )
            ],

            'layout': {
                'title': 'Number of different message types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message type"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=starting_names,
                    y=starting_counts
                )
            ],

            'layout': {
                'title': 'Number of sentences staring with a verb',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Type"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()