import pandas as pd

class TreeData(object):

    def __init__(self,filepath_to_df):
        self.df = self._get_df(filepath=filepath_to_df)

    def _get_df(self,filepath):
        return pd.read_pickle(filepath)

    def get_all(self):
        self.precision(self.df)
        self.recall(self.df)
        self.specificity(self.df)

    def precision(self,dataframe):
        ''' Get PRECISION '''
        precision = dataframe.tp.sum() / (dataframe.tp.sum()+dataframe.fp.sum())
        self._print_statement('precision', precision)
        return precision

    def recall(self,dataframe):
        ''' Get RECALL '''
        recall = dataframe.tp.sum() / (dataframe.tp.sum() + dataframe.fn.sum())
        self._print_statement('recall',recall)
        return recall

    def specificity(self,dataframe):
        '''Get SPECIFICITY '''
        specificity = dataframe.tn.sum() / (dataframe.tn.sum() + dataframe.fp.sum())
        self._print_statement('specificity',specificity)

    def _print_statement(self,metric,result):
        print('----- Your {} is {} -----'.format(metric.upper(),round(result,2)))

    def metric_by_zip(self):
        self.metric_by_block()
        zip_codes = self.df['zip_code'].unique().tolist()
        info = []

        for zippy_code in zip_codes:
            squares = []
            df = self.by_block[self.by_block['zip_code'] == zippy_code]
            n = df.shape[0]
            for block in df.iterrows():
                squares.append((block[1].predict_tree - block[1].label_tree)**2)
            MSE = round(sum(squares) / n,2)
            RMSE = round(MSE**0.5,2)
            df = self.df[self.df['zip_code']==[zippy_code]].groupby('zip_code',as_index=False).sum()
            for row in df.iterrows():
                precision,recall = self._precision_recall(row=row)
            #zip_code,precision,recall,rmse
            info.append([zippy_code,precision,recall,MSE,RMSE])
        self.metric_by_zip = pd.DataFrame(info,columns=['zip_code','precision','recall','mse','rmse'])

    def metric_by_block(self):
        info = []
        df = self.df.groupby('block',as_index=False).sum()
        for block in df.iterrows():
            zip_code = self.df[self.df.block == block[1].block]['zip_code'].iloc[0]

            # label_tree, predict_tree, precision,recall,zip code
            precision, recall = 0,0

            if (block[1].tp + block[1].fp) != 0:
                precision = block[1].tp / (block[1].tp + block[1].fp)

            if (block[1].tp + block[1].fn) != 0:
                recall = block[1].tp / (block[1].tp + block[1].fn)

            info.append([block[1].block,block[1].label, block[1].predicted, precision,recall,zip_code])
            self.by_block = pd.DataFrame(info,columns=['block','label_tree','predict_tree','precision','recall','zip_code'])

    def _precision_recall(self,row):
        precision,recall = 0,0

        if (row[1].tp + row[1].fp) != 0:
            precision = row[1].tp / (row[1].tp + row[1].fp)

        if (row[1].tp + row[1].fn) != 0:
            recall = row[1].tp / (row[1].tp + row[1].fn)

        return precision,recall

    def _RMSE(self,row,zippy_code):
        n = self.df[self.df.zip_code == zippy_code].shape[0]
        mean = self.df[self.df.zip_code == zippy_code].label.mean()








if __name__ == '__main__':
    trees = TreeData('data/predicted_test_pipeline.pkl')
    trees.metric_by_block()
    trees.metric_by_zip()
