# imports
import numpy as np
import pandas as pd
import pickle as pkl
import os
import openai
import sqlite3
import shap
import sklearn
import joblib
import gradio as gr
import re 
import editdistance


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from pandasql import sqldf
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from os.path import dirname, abspath

# importing functions
from requirement_2_funcs import get_dataframe_schema, query_to_filter, filter_to_sql, pipeline_extract, calc_SHAP, pred_and_stats
from requirement_3_funcs import get_importance_table, final_importance, ml_to_natural_language


# paths 
co2_dataset_queries_path = r'validation_files\co2_dataset_filter_repr_labels.txt'
co2_dataset_filter_repr_labels_path = r'validation_files\co2_dataset_filter_repr_labels.txt'
co2_dataset_sql_queries_labels_path = r'validation_files\co2_dataset_sql_code_labels.txt'

# openai api key, hard coded for now
openai_api_key = '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'

# extracting the dataset name
dataset_name = 'the_dataset'

# label name
label_name = 'CO2Emissions'

# Row ID
sample_ID = 'Country'

# pipeline
co2_dataset_pipeline_path = r'sustainability_model_dataset\co2_model.pkl'
with open(co2_dataset_pipeline_path, 'rb') as file:
    co2_pipeline = pkl.load(file)

# dataset
co2_dataset_path = r'sustainability_model_dataset\Sustainable_Energy_Invest_Forecast_Data_Reduced.csv'
co2_dataset = pd.read_csv(co2_dataset_path)

# extracting data context
schema_with_examples = get_dataframe_schema(co2_dataset, dataset_name)

# extracting model from pipeline and generating code
transformations, model = pipeline_extract(co2_pipeline)


# extract queries for loan dataset
with open(co2_dataset_queries_path, 'r') as file:
    co2_queries_list = file.readlines()
co2_queries_list = [line.strip() for line in co2_queries_list]


# extract filter representation labels
with open(co2_dataset_filter_repr_labels_path, 'r') as file:
    co2_dataset_filter_repr_labels_list = [line.rstrip('\n') for line in file]


# Open the file and read lines
with open(co2_dataset_sql_queries_labels_path, 'r') as file:
    co2_dataset_sql_queries_labels_lines = file.readlines()
# Initialize a list to store each SQL statement
co2_dataset_sql_queries_labels_list = []
# Process each line
for line in co2_dataset_sql_queries_labels_lines:
    # Split the line at \n
    parts = line.split('\\n')
    for part in parts:
        # Clean and add to the list
       co2_dataset_sql_queries_labels_list.append(part.strip())



# making predictions for filter representation
co2_dataset_filter_repr_predictions_list = []
for query in co2_queries_list:
    filter_representation = query_to_filter(openai_api_key, schema_with_examples, query)
    co2_dataset_filter_repr_predictions_list.append(filter_representation)


co2_dataset_sql_queries_predictions_list = []
for filter_repr in co2_dataset_filter_repr_labels_list:
    update_query, retrieve_query = filter_to_sql(openai_api_key, schema_with_examples, filter_repr)
    co2_dataset_sql_queries_predictions_list.append(update_query)
    co2_dataset_sql_queries_predictions_list.append(retrieve_query)



# exact match accuracy (EMA) and levenshtein distance for user query to filter representation translation
filter_repr_correct_count = 0
filter_repr_distance_list = []
for i in range(len(co2_dataset_filter_repr_predictions_list)):
    if co2_dataset_filter_repr_predictions_list[i] == co2_dataset_filter_repr_labels_list[i]:
        filter_repr_correct_count += 1

    filter_repr_distance = editdistance.eval(co2_dataset_filter_repr_predictions_list[i], co2_dataset_filter_repr_labels_list[i])
    filter_repr_distance_list.append(filter_repr_distance)

exact_match_accuracy_filter_repr = filter_repr_correct_count / len(co2_dataset_filter_repr_predictions_list)
average_levenshtein_distance_filter_repr = sum(filter_repr_distance_list) / len(filter_repr_distance_list)
print(f'EMA of Predictions for Filter Repr CO2 Dataset: {exact_match_accuracy_filter_repr}')
print(f'Avg Levenshtein Distance between Predictions and Labels for Filter Repr CO2 Dataset: {average_levenshtein_distance_filter_repr} ')
print('---------------------------------------------------------------------------------------------------------------')



# exact match accuracy (EMA) and levenshtein distance for user query to filter representation translation all lower case
filter_repr_correct_count_lower = 0
filter_repr_distance_list_lower = []
for i in range(len(co2_dataset_filter_repr_predictions_list)):
    if co2_dataset_filter_repr_predictions_list[i].lower() == co2_dataset_filter_repr_labels_list[i].lower():
        filter_repr_correct_count_lower += 1

    filter_repr_distance_lower = editdistance.eval(co2_dataset_filter_repr_predictions_list[i].lower(), co2_dataset_filter_repr_labels_list[i].lower())
    filter_repr_distance_list_lower.append(filter_repr_distance_lower)

exact_match_accuracy_filter_repr_lower = filter_repr_correct_count_lower / len(co2_dataset_filter_repr_predictions_list)
average_levenshtein_distance_filter_repr_lower = sum(filter_repr_distance_list_lower) / len(filter_repr_distance_list_lower)
print(f'EMA of Predictions for Filter Repr CO2 Dataset all lower: {exact_match_accuracy_filter_repr_lower}')
print(f'Avg Levenshtein Distance between Predictions and Labels for Filter Repr CO2 Dataset all lower: {average_levenshtein_distance_filter_repr_lower} ')
print('---------------------------------------------------------------------------------------------------------------')



# exact match accuracy (EMA) for filter representation translation to sql queries
sql_queries_correct_count = 0
sql_queries_distance_list = []
for i in range(len(co2_dataset_sql_queries_predictions_list)):
    if co2_dataset_sql_queries_predictions_list[i] == co2_dataset_sql_queries_labels_list[i]:
        sql_queries_correct_count += 1

    sql_queries_distance = editdistance.eval(co2_dataset_sql_queries_predictions_list[i], co2_dataset_sql_queries_labels_list[i])
    sql_queries_distance_list.append(sql_queries_distance)

exact_match_accuracy_sql_queries_loan = sql_queries_correct_count / len(co2_dataset_sql_queries_predictions_list)
average_levenshtein_distance_sql_queries = sum(sql_queries_distance_list) / len(sql_queries_distance_list)

print(f'EMA of Predictions for SQL Queries CO2 Dataset: {exact_match_accuracy_sql_queries_loan}')
print(f'Avg Levenshtein Distance between Predictions and Labels for SQL Queries CO2 Dataset: {average_levenshtein_distance_sql_queries} ')
print('---------------------------------------------------------------------------------------------------------------')