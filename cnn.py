# import theano
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard
import time
import matplotlib.pyplot as plt
from keras.models import load_model
from data_frame_operations import DFOps
from tree_analysis import TreeData


'''
Much credit to Keras, their excellent documentation and thorough tutorials
----- https://keras.io/getting-started/sequential-model-guide/ -----
'''


class NeuralNetwork(object):
    '''
    This class will train neural network on image data
    '''

    def __init__(self):
        '''
        -- Initialize class --
        '''
        pass

    def import_data(self,label_file_path,array_file_path,merge_on):
        '''
        -- Merge data into one array --

            PARAMETERS
            ----------
                label_file_path: str
                    File path to retrieve data containing filename, label in pickled pandas dataframe
                array_file_path: str
                    File path to retrive data containing filename and numpy array's of pictures
                merge_on: str
                    Column with pandas dataframe to merge on

            RETURNS
            -------
                None

            INITIALIZED
            -----------
                self.df:
                    Dataframe merged on 'merge_on'
        '''
        # pull data from arrays
        df_arrays = pd.read_pickle(array_file_path)
        df_labels = pd.read_pickle(label_file_path)
        self.df_test_pics = pd.read_pickle('test.pkl')
        self.test_array = self.df_test_pics['np_array']
        self.test_array = np.array(list(self.test_array.values))
        # merge df_arrays via left merge on merge_on
        self.df = df_arrays.merge(df_labels,how='left',on=[merge_on])

    def train_test_split(self,X_col_name,y_col_name,train_split):
        '''
        -- Merge data into one array --

            PARAMETERS
            ----------
                X_col_name: str
                    column name for X data
                y_col_name: str
                    column name for y data
                train_split: float -> 0.8
                    percent of data to be in train set
                    e.g. 80 percent train set, 20 percet test set
            RETURNS
            -------
                None

            INITIALIZED
            -----------
                self.X_train: np.array
                    array of pics to train on

                self.X_test: np.array
                    array of pics to test on
                self.y_train: np.array
                    array of labels to train on
                self.y_test:
                    array of labels to test on
        '''

        # set X and convert to array
        # X = self.df[X_col_name]
        X = self.df

        # set y
        y = self.df[y_col_name]

        # split into train,test data
        msk = np.random.rand(X.shape[0]) > train_split
        self.X_train_all = X[~msk]
        self.X_test_all = X[msk]
        self.y_train = y[~msk]
        self.y_test = y[msk]

        self.X_train = np.array(list(self.X_train_all[X_col_name].values))
        self.X_test = np.array(list(self.X_test_all[X_col_name].values))


    def set_parameters(self,
                        random_seed,
                        batch_size,
                        classes,
                        epochs,
                        image_dims,
                        num_filters,
                        pool,
                        kern_size,
                        colors):
        '''
        -- Set parameters for neural networks --

            See below for instantiated attribues and step by step explanations

            RETURNS
            -------
                None

            INSTANTIATED
            ------------
                See below
        '''


        # for reproducibility
        np.random.seed(random_seed)

        # see Keras documentation for explanations
        self.batch_size = batch_size
        self.nb_classes = classes
        self.nb_epoch = epochs

        # input image dimensions
        img_rows, img_cols = image_dims[0], image_dims[1]
        # number of convolutional filters to use
        self.nb_filters = num_filters
        # size of pooling area for max pooling
        self.pool_size = pool
        # convolution kernel size
        self.kernel_size = kern_size
        self.input_shape = (img_rows, img_cols, colors)

    def run_models(self):
        '''
        -- Train network --

            This method will need to be HARD CODED to edit model for training

            See below for step by step process

        '''
        start = time.time()
        # get data into correct format
        X_train = self.X_train.astype('float32')
        X_test = self.X_test.astype('float32')
        self.test_array = self.test_array.astype('float32')
        # normalize
        self.X_train_norm = X_train / 255
        self.X_test_norm = X_test / 255
        self.test_array /= 255
        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(self.y_train, self.nb_classes)
        Y_test = np_utils.to_categorical(self.y_test, self.nb_classes)
        # set model
        self.model = Sequential()
        # 2 convolutional layers followed by a pooling layer followed by dropout

        ''' -- Layer 1 -- '''
        self.model.add(Convolution2D(self.nb_filters, self.kernel_size[0], self.kernel_size[1],
                                border_mode='valid',
                                input_shape=self.input_shape))
        self.model.add(Activation('tanh'))
        # self.model.add(Dropout(0.25))

        ''' -- Layer 2 -- '''
        self.model.add(Convolution2D(self.nb_filters, self.kernel_size[0], self.kernel_size[1],
                                border_mode='valid',
                                input_shape=self.input_shape))
        self.model.add(Activation('tanh'))
        # self.model.add(Dropout(0.25))

        ''' -- Layer 3 -- '''
        self.model.add(MaxPooling2D(pool_size=self.pool_size))
        self.model.add(Dropout(0.25))
        # transition to an mlp
        self.model.add(Flatten())
        self.model.add(Dense(25))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(self.nb_classes))

        ''' -- Classification -- '''
        self.model.add(Activation('softmax'))
        #compile(self, optimizer, loss, metrics=None, sample_weight_mode=None, weighted_metrics=None)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        tbCallBack = TensorBoard(log_dir='./graph',
                                histogram_freq=None,
                                write_graph=True,
                                write_images=True,
                                embeddings_metadata=True)

        # self.model.fit(self.X_train_norm, Y_train, batch_size=self.batch_size, epochs=self.nb_epoch,
        #           verbose=1, validation_data=(self.X_test_norm, Y_test),callbacks=[tbCallBack],
        #           class_weight='auto')

        self.model.fit(self.X_train_norm, Y_train, batch_size=self.batch_size, epochs=self.nb_epoch,
                  verbose=1, validation_split=0.2,callbacks=[tbCallBack],
                  class_weight='auto')
        self.score = self.model.evaluate(self.X_test_norm, Y_test, verbose=0)
        end_time = time.time()-start
        day = time.ctime().lower().replace(' ','_')
        print('Test score:', self.score[0])
        print('Test accuracy:', self.score[1])
        print('Total time to run: {}'.format(int(end_time/60)))
        self.model.save('models/{}_{}'.format(round(self.score[1],4),day))

    def output_predictions(self):
        predicted = self.model.predict_classes(self.X_test_norm)
        predicted = predicted.tolist()
        df_predicted = self.X_test_all
        df_predicted['predicted'] = predicted
        self.df_predicted = df_predicted.drop('np_array',axis=1)
        self.filename = 'data/predicted_{}.pkl'.format(time.ctime().replace(' ','_'))
        df_predicted.to_pickle('{}'.format(self.filename))

if __name__ == '__main__':
    NN = NeuralNetwork()
    NN.import_data(label_file_path='data/labeled.pkl',
                    array_file_path='data/resized.pkl',
                    merge_on='filename')

    NN.train_test_split(X_col_name='np_array',
                        y_col_name='label',
                        train_split=0.8)

    NN.set_parameters(random_seed=17,
                        batch_size=10,
                        classes=10,
                        epochs=10,
                        image_dims=(100,50),
                        num_filters=5,
                        pool = (3, 3),
                        kern_size = (3, 3),
                        colors=3)


    NN.run_models()
    NN.output_predictions()
    data = DFOps(filepath_to_df=NN.filename)
    data.perform_all_ops()
    data.df.to_pickle('test')
    trees = TreeData('test','data/labeled_edited.pkl')
    trees.get_all()
    print(trees.tfRMSE())
