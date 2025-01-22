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

# openai api key, hard coded for now
openai_api_key = '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'

# loading the model pipeline and dataset, will be replaced with interface
# pipeline, hard coded for now
pipeline_path = r'sustainability_model_dataset\sustainability_model.pkl'
with open(pipeline_path, 'rb') as file:
    pipeline = pkl.load(file)

# dataset, hard coded for now
dataset_path = r'sustainability_model_dataset\Sustainable_Energy_Invest_Forecast_Data_Reduced.csv'
dataset = pd.read_csv(dataset_path)

# extracting the dataset name
dataset_name = 'the_dataset'

# label name
label_name = 'CO2Emissions'

# Row ID
sample_ID = 'Country'

# user query, hard coded for now
# What is your reasoning for deciding if people with no missed payments are good credit risk?
# How many people happened to ask for loans worth less than 2000?
# If we were to increase the loan amount by 250, what would happen to the likelihood of being bad credit risk for the data point with id 993
# If people in the data were unemployed, what are important features for predicting credit risk?
# How likely are the people older than 30 to be good credit risk?
# Which three features are the most important for the model's predictions in the data?
# Assuming that the age of unemployment is decreased by 5 years, what would the prediction be?
# Are people younger than 25 vulnerable to bad credit risk?
# Name the three most important features to determine whether those that are applying for furniture loans are good credit risk.
# Why did the model predict data point number 993 and is there anything you can do to change it?
user_query = 'What are the most important features affecting CO2 emissions'

# getting datacontext
data_context = get_dataframe_schema(dataset, dataset_name)

# extracting model from pipeline and generating code, this needs to be run only once at the beginning
transformations, model = pipeline_extract(pipeline)

# getting filter representation
filter_representation = query_to_filter(openai_api_key, data_context, user_query)
print('---------------------------------')
print(filter_representation)

# getting sql code
update_query, retrieve_query = filter_to_sql(openai_api_key, data_context, filter_representation)
print('---------------------------------')
print(update_query)
print(retrieve_query)

# getting SHAP values and relevant datasets
before_dataset, shap_values_before, after_dataset, shap_values_after = calc_SHAP(transformations, model, dataset, dataset_name, update_query, retrieve_query, label_name, sample_ID)

# getting final predictions and aggregate statistics
final_dataset_before, summary_stats_before, final_dataset_after, summary_stats_after = pred_and_stats(pipeline, before_dataset, after_dataset, label_name, sample_ID)

# getting relevant importance tables
importance_table_before, importance_table_after, feature_names = final_importance(shap_values_before, shap_values_after, final_dataset_after, label_name, sample_ID)

# getting the final response
response = ml_to_natural_language(shap_values_before, shap_values_after, feature_names, user_query, filter_representation, final_dataset_before, final_dataset_after, importance_table_before, importance_table_after, summary_stats_after, openai_api_key, label_name, sample_ID)
print('---------------------------------')
print(response)