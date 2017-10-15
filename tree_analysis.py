import pandas as pd

class TreeData(object):

    def __init__(self,filepath_to_df,filepath_to_labeled):
        self.df, self.df_all_labeled = self._get_df(filepath=filepath_to_df,filepath_all_labeled=filepath_to_labeled)


    def _get_df(self,filepath,filepath_all_labeled):
        return pd.read_pickle(filepath), pd.read_pickle(filepath_all_labeled)

    def get_all(self):
        self.precision(self.df), self.recall(self.df)


    def precision(self,dataframe):
        ''' -- Get PRECISION -- '''
        self.precision = dataframe.tp.sum() / (dataframe.tp.sum()+dataframe.fp.sum())
        self._print_statement('precision', self.precision)
        return self.precision

    def recall(self,dataframe):
        ''' -- Get RECALL -- '''
        self.recall = dataframe.tp.sum() / (dataframe.tp.sum() + dataframe.fn.sum())
        self._print_statement('recall',self.recall)
        return self.recall

    def specificity(self,dataframe):
        ''' -- Get SPECIFICITY -- '''
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
            info.append([zippy_code,precision,recall,RMSE])
        self.metric_by_zip = pd.DataFrame(info,columns=['zip_code','precision','recall', 'rmse'])

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

    def avg_trees_per_block(self):
        df = self.df.groupby('block', as_index=False).sum()
        return round((df.label.sum() / df.shape[0]),1)

    def all_data_avg_trees_per_block(self):
        df = self.df_all_labeled.groupby('block', as_index=False).sum()
        return round((df.label.sum() / df.shape[0]),1)


    def tfRMSE(self):
        df = self.df.groupby('block', as_index=False).sum()
        tree_factor = df.label.sum()
        squares = []
        for row in df.iterrows():
            tree_factor_residual = ((row[1].predicted - row[1].label)**2)*(row[1].label/tree_factor)
            squares.append(tree_factor_residual)
        tfRMSE = sum(squares)**(1/2)
        self._print_statement('RMSE',tfRMSE)
        return tfRMSE

    def f1_score(self):
        self.f1_score = 2*(1/(1/self.precision + 1/self.recall))
        self._print_statement('f1 Score',self.f1_score)
        return self.f1_score




if __name__ == '__main__':
    trees = TreeData('data/predicted_test_pipeline.pkl')
    trees.metric_by_block()
    trees.metric_by_zip()
