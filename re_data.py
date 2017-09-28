import pyspark as ps    # for the pyspark suite
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType
import string
import unicodedata

class RE_EDA(object):
    def __init__(self,filepath):
        self.spark = ps.sql.SparkSession.builder \
                    .master("local[4]") \
                    .appName("re_eda") \
                    .getOrCreate()
        self.df = self._load_data(filepath)



    def _load_data(self,filepath):
        return self.spark.read.format(filepath).option("header", "true")


if __name__ == '__main__':
    re = RE_EDA('data/opa_public.csv')
