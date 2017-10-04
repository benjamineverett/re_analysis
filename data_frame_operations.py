import pandas as pd
import numpy as np

class DFOps(object):
    def __init__(self,filepath_to_df):
        self.df = self._get_df(filepath=filepath_to_df)

    def perform_all_ops(self):
        self.drops()
        self.create_results()
        self.add_zip()
        
    def _get_df(self,filepath):
        return pd.read_pickle(filepath)

    def drops(self):
        self.df = self.df.drop('np_array',axis=1)
        self.df.reset_index(inplace=True,drop=True)

    def create_results(self):
        # create True Positive
        self.df['tp'] = np.where((self.df.label == 1) & (self.df.predicted == 1),1,0)
        # create True Negative
        self.df['tn'] = np.where((self.df.label == 0) & (self.df.predicted == 0),1,0)
        # create False Positive
        self.df['fp'] = np.where((self.df.label == 0) & (self.df.predicted == 1),1,0)
        # create False Negative
        self.df['fn'] = np.where((self.df.label == 1) & (self.df.predicted == 0),1,0)
        # change order of columns
        self.df = self.df.reindex_axis(['filename','block','predicted','label','tp','fp','tn','fn'],axis=1)

    def add_zip(self):
        self.df['zip_code'] = self.df.filename.apply(lambda x: x.split('_')[-2])
        self.df['zip_code'] = self.df['zip_code'].astype('int')

    def to_pickle(self,new_file_name):
        self.df.to_pickle(new_file_name)






if __name__ == '__main__':
    ops = DFOps('data/predicted_on_test.pkl')
    ops.drops()
    ops.create_results()
    ops.add_zip()
    ops.to_pickle(new_file_name='data/predicted_test_pipeline.pkl')
