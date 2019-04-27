import pandas as pd
import numpy as np
import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import string
import unicodedata
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
import re
from pyspark.conf import SparkConf
from pyspark.sql import functions as F
import string
import csv
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType
import tarfile
from six.moves import urllib
from config_file import DOWNLOAD_ROOT, DATASETS_PATH, DOWNLOAD_URL, TWEETS_PATH, TRAINING_DATA_URL



spark = SparkSession.builder.master("local[*]").appName("nlp").config("spark.executor.memory", "32g")    .config("spark.driver.memory", "32g").config("spark.memory.offHeap.enabled",True)     .config("spark.memory.offHeap.size","16g").config("spark.debug.maxToStringFields","200").getOrCreate()


def fetch_data(tgz_name, download_url=DOWNLOAD_URL, datasets_path=DATASETS_PATH):
    if not os.path.isdir(datasets_path):
        os.makedirs(datasets_path)
    tgz_path = os.path.join(datasets_path, tgz_name)
    urllib.request.urlretrieve(download_url, tgz_path)
    tweeter_tgz = tarfile.open(tgz_path)
    tweeter_tgz.extractall(path=datasets_path)
    tweeter_tgz.close()
    
def fetch_training_data(download_url=TRAINING_DATA_URL, datasets_path=DATASETS_PATH):
    if not os.path.isdir(datasets_path):
        os.makedirs(datasets_path)
    tgz_path = os.path.join(datasets_path, "trainingandtestdata.zip")
    urllib.request.urlretrieve(download_url, tgz_path)
    train_tgz = tarfile.open(tgz_path)
    train_tgz.extractall(path=datasets_path)
    train_tgz.close()
                        
def load_data(path, join):
    csv_path = os.path.join(path,join)
    return spark.read.csv(csv_path, inferSchema=True, encoding = 'utf8', header=True)

# Remove https in the text
def remove_https(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    return text

def normalizeData(text):    
    text = unicodedata.normalize('NFKD', str(text))
    text = text.replace(r'\n', '')
    text = ' '.join(text.split())
    replace_punctuation = str.maketrans(string.punctuation,' '*len(string.punctuation))
    text = text.translate(replace_punctuation)
    text = text.encode('ASCII', 'ignore')
    text = text.decode('unicode_escape')
    text = ''.join([i for i in text if not i.isdigit()])
    return text


def clean_latin1(data):
    for _hex, _char in LATIN_1_CHARS:
        data = data.replace(_hex, _char)
    return data

def lower_words(df):
    fields = df.schema.fields
    stringFields = filter(lambda f: isinstance(f.dataType, StringType), fields)
    nonStringFields = map(lambda f: col(f.name), filter(lambda f: not isinstance(f.dataType, StringType), fields))
    stringFieldsTransformed = map(lambda f: lower(col(f.name)).alias(f.name), stringFields) 
    allFields = [*nonStringFields, *stringFieldsTransformed]
    df = df.select(allFields)
    return df

def regexTokenizer_StopWordsRemover(df, inputCol, outputCol1, outputCol2, ):
    # the input od the stop words is the outputCol1 from tokenizer
    # inputCol and outputCol should be in string format
    regexTokenizer = RegexTokenizer(minTokenLength= 3, inputCol=inputCol, outputCol=outputCol1, pattern="\\W")
    df = regexTokenizer.transform(df)
    remover = StopWordsRemover(inputCol=outputCol1, outputCol=outputCol2)
    df = remover.transform(df).drop(inputCol, outputCol1)
    return df

def remove_emoji(df, inputCol, outputCol):
    with open(os.path.join(DATASETS_PATH,'emoji.txt'), 'r') as f:
        first_list=f.read().strip().splitlines()

    second_list = []
    for item in first_list:
        x = item.split('\\')
        second_list.append(x)

    third_list = []
    for item in second_list:
        new_list = []
        for e in item:
            if e not in (''):
                new_list.append(e)
        third_list.append(new_list)

    fourth_list = [item for sublist in third_list for item in sublist]

    emojies = []
    for word in fourth_list:   #for each word in line.split()
        if word not in emojies:    #if a word isn't in line.split            
            emojies.append(word.lower())

    remover = StopWordsRemover(inputCol=inputCol, outputCol=outputCol, stopWords=emojies)
    df = remover.transform(df).drop(inputCol).withColumnRenamed(outputCol, inputCol)
    return df

def change_city_with_country(df, nameCol='location'):
    
    list_city_country=load_data(DATASETS_PATH,"GeoLite2-City-Locations-en.csv")
    list_city_country.drop_duplicates(["city_name"])
    list_city_country=list_city_country.filter("city_name is not NULL")

    list_city = [i.city_name for i in list_city_country.collect()]
    list_country = [i.country_name for i in list_city_country.collect()]

    list_city = [''.join(c for c in s if c not in string.punctuation) for s in list_city]
    list_country = [''.join(c for c in s if c not in string.punctuation) for s in list_country]


    list_city = [x.lower() for x in list_city]
    list_country = [x.lower() for x in list_country]
    
    df_pd = df.toPandas()
    df_pd = df_pd[df_pd[nameCol].isin(list_city)]
    
    for pos, word in enumerate(list_city):
        df_pd.loc[df_pd.location == word, 'location'] = list_country[pos]
    
    sqlCtx = SQLContext(sc)
    df=sqlCtx.createDataFrame(df_pd)
    return df

def avg_happiness_of_words(df,nameCol_to_explode):
    h_avg = sc.textFile(os.path.join(DATASETS_PATH, "Data_Set_S1.txt")).mapPartitions(lambda line:csv.reader(line,delimiter='\t', quotechar='"')).filter(lambda line: len(line)>=2 and line[0]!= 'word')           .toDF(["word", "happiness_rank", "happiness_average","happiness_standard_deviation", "twitter_rank",                  "google_rank", "nyt_rank", "lyrics_rank"])

    happiness_words = h_avg.select('word').collect()
    happiness_average = h_avg.select('happiness_average').collect()
    happiness_words = [str(row.word) for row in happiness_words]
    happiness_average = [float(row.happiness_average) for row in happiness_average]
    
    nameCols = df.columns
    df=df.withColumn('row_index', F.monotonically_increasing_id()).select(["row_index"]+nameCols)
    
    df_explode = df.withColumn("elements", explode(nameCol_to_explode))
    df_explode_pd = df_explode.toPandas()
    
    for pos, word in enumerate(happiness_words):
        if happiness_average[pos] <= 3.5 or happiness_average[pos] >=6.5 :
            df_explode_pd.loc[df_explode_pd.elements == word, 'happiness_score'] = happiness_average[pos]
    
    df_explode_pd_n = df_explode_pd.dropna()
    
    sqlCtx = SQLContext(sc)
    df_explode=sqlCtx.createDataFrame(df_explode_pd_n)
        
    df_ = df_explode.orderBy("row_index").groupBy("row_index" ).agg(collect_list("happiness_score"))
    df_ = df_.select(col("row_index").alias("row_index"),                         col("collect_list(happiness_score)").alias("happiness_score"))
    
    df_join = df.join(df_, on='row_index', how = 'inner').sort('row_index').drop('row_index')
    
    def mean_list(x):
        summation = 0.00
        for val in x:
            
            summation = float(summation) + float(val)
        return [float(summation)/float(len(x))]

    mean_list_udf = udf(lambda y: mean_list(y), ArrayType(FloatType()))
    
    df = df_join.withColumn('happiness_avg', mean_list_udf('happiness_score'))
    df_final= df.withColumn("happiness_avg", explode('happiness_avg'))

    return df_final

def remove_irrelevant_words(df,nameCol_to_explode):
    h_avg = sc.textFile(os.path.join(DATASETS_PATH, "Data_Set_S1.txt"))\
           .mapPartitions(lambda line: csv.reader(line,delimiter='\t', quotechar='"'))\
           .filter(lambda line: len(line)>=2 and line[0]!= 'word')\
           .toDF(["word", "happiness_rank", "happiness_average",\
                  "happiness_standard_deviation", "twitter_rank",\
                  "google_rank", "nyt_rank", "lyrics_rank"])
    
    # the list of words in Amazon Turk
    list_words = h_avg.select('word').collect()
    list_words = [str(row.word) for row in list_words]
    #----------------------------------------------------------
    
    nameCols = df.columns
    df=df.withColumn('row_index', F.monotonically_increasing_id()).select(["row_index"]+nameCols)
    
    df_explode = df.withColumn("happiness_words", explode(nameCol_to_explode))
    df_explode = df_explode.where(df_explode.happiness_words.isin(list_words))
    
    df_explode = df_explode.dropna()

        
    df_explode = df_explode.orderBy("row_index").groupBy("row_index" ).agg(collect_list('happiness_words'))
    df_explode = df_explode.select(col("row_index").alias("row_index"),\
                                   col("collect_list(happiness_words)").alias("happiness_words"))
    
    df = df.join(df_explode, on='row_index', how = 'inner').\
                                sort('row_index').drop('row_index', nameCol_to_explode)
    return df

# making avarage over the whole tweets for one location ignorinf the tweets with 
# happiness avarge between 4 and 6
def avg_happiness_over_country(df, groupby_col, happiness_avg_over_words_col):
    df = df.groupBy(groupby_col).agg(collect_list(happiness_avg_over_words_col).alias("avg_happiness_country"))
    def mean_list(x):
        summation = 0.00
        for val in x:
            if val< 4 or val > 6:
                summation = float(summation) + float(val)
        return [float(summation)/float(len(x))]

    mean_list_udf = udf(lambda y: mean_list(y), ArrayType(FloatType()))

    df = df.withColumn('avg_happiness_country', mean_list_udf('avg_happiness_country'))
    df = df.withColumn("avg_happiness_country", explode('avg_happiness_country'))
    
    return df

def avg_happiness_over_date(df, groupby_col, happiness_avg_over_words_col):
    nameCols = df.columns
    #df = df.withColumn('row_index', F.monotonically_increasing_id()).select(['row_index']+nameCols)
    df = df.groupBy(groupby_col).agg(collect_list(happiness_avg_over_words_col).\
                                     alias("happiness_avg"))
    def mean_list(x):
        summation = 0.00
        for val in x:
            #if val< 4 or val > 6:
            summation = float(summation) + float(val)
        return [float(summation)/float(len(x))]

    mean_list_udf = udf(lambda y: mean_list(y), ArrayType(FloatType()))

    df = df.withColumn('happiness_avg', mean_list_udf('happiness_avg'))
    df = df.withColumn("happiness_avg", explode('happiness_avg'))
    
    return df


