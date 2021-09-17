import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    #read in messages .csv
    message = pd.read_csv(messages_filepath)
    
    #read in cateogries .csv
    categories = pd.read_csv(categories_filepath)
    
    #split the strings in the 'categories' column, and expand into a big dataframe
    categories = categories.categories.str.split(';',expand=True)
    
    #select the first row, and strip, from the right-hand side, the '-1' and '0-'. Make this the column names
    row = categories.iloc[0]
    category_colnames = row.map(lambda x: x.rstrip('-1,-0'))
    categories.columns = category_colnames
    
    #loop through columns of the categories dataframe
    for column in categories:
        #take the last letter (number) of the string
        categories[column] = categories[column].str[-1]
        #convert the string into an integer
        categories[column] = categories[column].astype(int)
    
    #concatenate categories and messages on 'id' and return
    return pd.merge(message, categories,left_index=True,right_index=True)
    


def clean_data(df):

    # replace '2's with '1's in columns
    for col in df.columns:
        column = df[col]
        df[col] = column.replace(0,1)
    
    # return a dataframe without the duplacates
    return df[~df.duplicated()]


def save_data(df, database_filename):
    # saves the data to an sql database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('data', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()