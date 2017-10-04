import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error, r2_score

'''
NOT A WORKING MODEL
Code herein contains scraps of beginnings to analyze Philly real estate data
'''

class Models(object):
    '''
    Models analyzes real estate data using specified models
    '''

    def __init__(self,model_type,filename):
        '''
        Initialize class

            PARAMETERS
            ----------
                model_type: sklearn model_type
                    see sklean documentation for different models
                filename: str
                    path to file to import data
        '''
        self.model_type = model_type
        self._initialize(filename)

    def _initialize(self,filename):
        '''
        Initialize data

            PARAMETERS
            ----------
                filename: str
                    from self.__init__

            INITIALIZES
            -----------
                self.df: pandas dataframe
                self.get_info(): display helpful info off top
        '''
        # get data
        self.df = pd.read_pickle(filename)
        # show a few things off the basements
        self.get_info(dataframe=self.df)

    def get_info(self, dataframe):
        print('\n')
        print('---------- COLUMNS ----------')
        print(dataframe.columns)
        print('\n')
        print('---------- SHAPE ----------')
        print(dataframe.shape)
        print('\n')
        print('---------- NULL VALUES ----------')
        print(dataframe.isnull().sum())
        print('\n\n')
        print('---------- DATA TYPES ----------')
        print(dataframe.dtypes)
        print('\n\n')
        print('---------- INFO ----------')
        print(dataframe.info())

    def train_test_split(self,X,y,random_state=27,shuffle=True,test_size=0.2):
        X = normalize(X)
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,y)
        print('X_train shape: {}'.format(self.X_train.shape))
        print('y_train shape: {}'.format(self.y_train.shape))
        print('X_test shape: {}'.format(self.X_test.shape))
        print('y_test shape: {}'.format(self.y_test.shape))

    def fit(self, X_train, y_train, n_jobs=-1):
        '''
        Fit model on training data

            PARAMETERS
            ----------
                X_train: np_array, pandas dataframe
                    set to train on
                y_train: np_array, pandas dataframe
                    predictions to train on

            INITIALIZES
            -----------

                self.model: sklearn model using self.model_type

            '''

        self.model = self.model_type()
        self.model = self.model.fit(X_train,y_train)

    def predict(self):
        '''
        Predict on test data
        '''
        self.y_predicted = self.model.predict(self.X_test)

if __name__ == '__main__':
    re = Models(model_type=LinearRegression,filename='data/fairmount.pkl')
    re.fairmount = re.df[(re.df.sale_year > 2008) & (re.df.zip_code == 19130) & (re.df.sale_price > 100000)]
    re.y = re.fairmount.pop('sale_price')
    re.X = re.fairmount[['number_of_bathrooms','number_of_bedrooms',
                        'number_of_rooms','number_stories','sale_year',
                        'total_area','garage_spaces','depth',
                        'frontage', 'exterior_condition']]
    re.train_test_split(re.X,re.y)
