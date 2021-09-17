import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
nltk.download('averaged_perceptron_tagger')
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    # this function loads in the data and splits into X and Y
    
    # create an engine object in sqlalchemy, and load in the database
    engine = create_engine('sqlite:///' + database_filepath)
    
    # turn the table 'data' table into a dataframe, and split into X and Y
    df = pd.read_sql("SELECT * FROM main", engine)
    
    # get the category names
    categories = list(set(df.columns).difference(set(['id', 'message', 'original', 'genre'])))
    
    X = df.message
    Y = df[categories]
    
    return X,Y,categories


def tokenize(text):
    # this function tokenizes text, normalizes, removes stopwords, and reduces words to their most basic form
    
    # normalize
    text = text.lower() 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    # tokenize
    text = word_tokenize(text)

    # remove stop words
    text = [w for w in text if w not in stopwords.words("english")]

    # reduce words to their root form
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
    
    return text

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

def build_model():
    #pipeline = Pipeline([
    #    ('features', FeatureUnion([('text_pipeline', Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
    #                                                           ('tfidf', TfidfTransformer()) ])), 
    #                              ('starting_verb', StartingVerbExtractor()) ])), 
    #    ('clf', MultiOutputClassifier(RandomForestClassifier())) ]) 
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',  MultiOutputClassifier(RandomForestClassifier()))
     ])

    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000),
        #'clf__n_estimators': [50, 100, 200],
        #'clf__min_samples_split': [2, 3, 4],

    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv




def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    for i,colname in enumerate(category_names):
        pred = [line[i] for line in Y_pred]
        true = Y_test.iloc[:,i]
        print('\033[1m' + colname)
        print('\033[0m' +classification_report(true, pred, labels=[0, 1]))



def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()