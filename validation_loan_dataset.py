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
loan_dataset_queries_path = r'validation_files\loan_dataset_queries.txt'
loan_dataset_filter_repr_labels_path = r'validation_files\loan_dataset_filter_repr_labels.txt'
loan_dataset_sql_queries_labels_path = r'validation_files\loan_dataset_sql_queries_labels.txt'

# openai api key, hard coded for now
openai_api_key = ''

# extracting the dataset name
dataset_name = 'the_dataset'

# label name
label_name = 'y'

# Row ID
sample_ID = 'ID'

# pipeline
loan_dataset_pipeline_path = r'loan_model_dataset\german_model_grad_tree.pkl'
with open(loan_dataset_pipeline_path, 'rb') as file:
    loan_pipeline = pkl.load(file)

# dataset
loan_dataset_path = r'loan_model_dataset\german_test.csv'
loan_dataset = pd.read_csv(loan_dataset_path)

# extracting data context
schema_with_examples = get_dataframe_schema(loan_dataset, dataset_name)

# extracting model from pipeline and generating code
transformations, model = pipeline_extract(loan_pipeline)


# extract queries for loan dataset
with open(loan_dataset_queries_path, 'r') as file:
    loan_queries_list = file.readlines()
loan_queries_list = [line.strip() for line in loan_queries_list]


# extract filter representation labels
with open(loan_dataset_filter_repr_labels_path, 'r') as file:
    loan_dataset_filter_repr_labels_list = [line.rstrip('\n') for line in file]


# extract sql queries labels
# open the file and read lines
with open(loan_dataset_sql_queries_labels_path, 'r') as file:
    loan_dataset_sql_queries_labels_lines = file.readlines()
# Initialize a list to store each SQL statement
loan_dataset_sql_queries_labels_list = []
# Process each line
for line in loan_dataset_sql_queries_labels_lines:
    # Split the line at \n
    parts = line.split('\\n')
    for part in parts:
        # Clean and add to the list
        loan_dataset_sql_queries_labels_list.append(part.strip())



# making predictions for filter representation
loan_dataset_filter_repr_predictions_list = []
for query in loan_queries_list:
    filter_representation = query_to_filter(openai_api_key, schema_with_examples, query)
    loan_dataset_filter_repr_predictions_list.append(filter_representation)


loan_dataset_sql_queries_predictions_list = []
for filter_repr in loan_dataset_filter_repr_labels_list:
    update_query, retrieve_query = filter_to_sql(openai_api_key, schema_with_examples, filter_repr)
    loan_dataset_sql_queries_predictions_list.append(update_query)
    loan_dataset_sql_queries_predictions_list.append(retrieve_query)


# exact match accuracy (EMA) and levenshtein distance for user query to filter representation translation
filter_repr_correct_count = 0
filter_repr_distance_list = []
for i in range(len(loan_dataset_filter_repr_predictions_list)):
    if loan_dataset_filter_repr_predictions_list[i] == loan_dataset_filter_repr_labels_list[i]:
        filter_repr_correct_count += 1

    filter_repr_distance = editdistance.eval(loan_dataset_filter_repr_predictions_list[i], loan_dataset_filter_repr_labels_list[i])
    filter_repr_distance_list.append(filter_repr_distance)

exact_match_accuracy_filter_repr = filter_repr_correct_count / len(loan_dataset_filter_repr_predictions_list)
average_levenshtein_distance_filter_repr = sum(filter_repr_distance_list) / len(filter_repr_distance_list)
print(f'EMA of Predictions for Filter Repr Loan Dataset 0 eg: {exact_match_accuracy_filter_repr}')
print(f'Avg Levenshtein Distance between Predictions and Labels for Filter Repr Loan Dataset 0 eg: {average_levenshtein_distance_filter_repr} ')
print('---------------------------------------------------------------------------------------------------------------')



# exact match accuracy (EMA) and levenshtein distance for user query to filter representation translation all lower case
filter_repr_correct_count_lower = 0
filter_repr_distance_list_lower = []
for i in range(len(loan_dataset_filter_repr_predictions_list)):
    if loan_dataset_filter_repr_predictions_list[i].lower() == loan_dataset_filter_repr_labels_list[i].lower():
        filter_repr_correct_count_lower += 1

    filter_repr_distance_lower = editdistance.eval(loan_dataset_filter_repr_predictions_list[i].lower(), loan_dataset_filter_repr_labels_list[i].lower())
    filter_repr_distance_list_lower.append(filter_repr_distance_lower)

exact_match_accuracy_filter_repr_lower = filter_repr_correct_count_lower / len(loan_dataset_filter_repr_predictions_list)
average_levenshtein_distance_filter_repr_lower = sum(filter_repr_distance_list_lower) / len(filter_repr_distance_list_lower)
print(f'EMA of Predictions for Filter Repr Loan Dataset all lower 0 eg: {exact_match_accuracy_filter_repr_lower}')
print(f'Avg Levenshtein Distance between Predictions and Labels for Filter Repr Loan Dataset all lower 0 eg: {average_levenshtein_distance_filter_repr_lower} ')
print('---------------------------------------------------------------------------------------------------------------')



# exact match accuracy (EMA) for filter representation translation to sql queries
sql_queries_correct_count = 0
sql_queries_distance_list = []
for i in range(len(loan_dataset_sql_queries_predictions_list)):
    if loan_dataset_sql_queries_predictions_list[i] == loan_dataset_sql_queries_labels_list[i]:
        sql_queries_correct_count += 1

    sql_queries_distance = editdistance.eval(loan_dataset_sql_queries_predictions_list[i], loan_dataset_sql_queries_labels_list[i])
    sql_queries_distance_list.append(sql_queries_distance)

exact_match_accuracy_sql_queries_loan = sql_queries_correct_count / len(loan_dataset_sql_queries_predictions_list)
average_levenshtein_distance_sql_queries = sum(sql_queries_distance_list) / len(sql_queries_distance_list)

print(f'EMA of Predictions for SQL Queries Loan Dataset 0 eg: {exact_match_accuracy_sql_queries_loan}')
print(f'Avg Levenshtein Distance between Predictions and Labels for SQL Queries Loan Dataset 0 eg: {average_levenshtein_distance_sql_queries} ')
print('---------------------------------------------------------------------------------------------------------------')