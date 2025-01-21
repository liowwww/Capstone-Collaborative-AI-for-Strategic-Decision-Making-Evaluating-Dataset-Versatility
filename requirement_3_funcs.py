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
from pandasql import sqldf
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from os.path import dirname, abspath


def get_importance_table(shap_values, feature_names):
    """
    Calculate the mean absolute SHAP values for each feature and return a sorted DataFrame of feature importance.
    
    Parameters:
    shap_values (np.ndarray): A numpy array of SHAP values (n_samples x n_features).
    feature_names (list or np.ndarray): A list or array of feature names corresponding to the SHAP values.
    
    Returns:
    pd.DataFrame: A DataFrame containing feature names and their corresponding importance, sorted in descending order.
    """
    # Calculate the mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create a DataFrame to store the feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    }).sort_values(by='Importance', ascending=False)
    
    return importance_df


# function to get relevant importance tables
def final_importance(shap_values_before, shap_values_after, final_dataset_after):

    # extracting the feature names
    shap_frame = final_dataset_after.drop(columns = ['ID','prediction','y'])
    feature_names = list(shap_frame.columns)

    if shap_values_before is None:
        importance_table_before = None
        importance_table_after = get_importance_table(shap_values_after,feature_names)
        return importance_table_before, importance_table_after, feature_names
    else:
        
        importance_table_before = get_importance_table(shap_values_before,feature_names)
        importance_table_after = get_importance_table(shap_values_after,feature_names)
        return importance_table_before, importance_table_after, feature_names
    

# function for ml output to natural language
def ml_to_natural_language(shap_values_before, shap_values_after, feature_names, user_query, filter_representation, final_dataset_before, final_dataset_after, importance_table_before, importance_table_after, summary_stats_after, openai_api_key):

    # adjusting the filter representation
    final_filter_representation = filter_representation.replace("[E]", "").strip()

    # making a dataframe with column names out of SHAP values
    shap_table_before = pd.DataFrame(shap_values_before, columns=feature_names)
    shap_table_after = pd.DataFrame(shap_values_after,columns = feature_names)

    response_prompt = PromptTemplate(
    input_variables=[
            "user_query", "final_filter_representation", "shap_values_before", 
            "shap_values_after", "final_dataset_before", "final_dataset_after", 
            "importance_table_before", "importance_table_after", "summary_stats_after"
        ],
        template=(
            "You are an assistant designed to help investors analyze and explain ESG (Environmental, Social and Governance) Risk Scores."  

            "Your job is to provide clear, concise, and accurate answers strictly based on the provided data. " 

            "Do not make assumptions or use external knowledge beyond the given inputs.\n\n" 

             

            "### Instructions:\n" 

            "- Provide an answer strictly based on the filter representation: {filter_representation}.\n" 

            "- Use the input data to generate insights that align with this representation.\n" 

            "- If the required data is missing or insufficient to answer the query, state: 'The data provided is insufficient to answer this query.'\n" 

            "- Do not provide explanations about the process of finding the answer.\n" 

            "- Avoid introducing external information, context, or general knowledge into your response.\n\n" 

             

            "### Dataset and Model Explanation:\n" 

            "The dataset contains information of companies listed in the S&P 500 index, including various features about their sector, employees, and other factors" 

            "The dataset includes a prediction of their ESG risk score based on the column 'prediction':\n" 

            "- The higher the ESG Risk score, the less attractive that company becomes to invest in.\n\n" 

             

            "### Relevant Data:\n" 

            "Here is the relevant information from the user query: {filter_representation}\n\n" 

            "SHAP values for the original dataset (before any updates, use if applicable): {shap_values_before}\n\n" 

            "Original dataset (use if applicable): {final_dataset_before}\n\n" 

            "SHAP values for the updated dataset: {shap_values_after}\n\n" 

            "Updated dataset: {final_dataset_after}\n\n" 

            "Feature importance represented by mean absolute SHAP values (original dataset, use if applicable): {importance_table_before}\n\n" 

            "Feature importance represented by mean absolute SHAP values (updated dataset): {importance_table_after}\n\n" 

            "Summary Table of the dataset to assist in providing statistics such as mean, min, max, etc: {summary_stats_after}\n\n" 

             

            "### Notes:\n" 

            "- Your answers must come *only* from the input variables.\n" 

            "- If data is missing or insufficient to answer the query, clearly state this instead of inferring.\n" 

            "- Do not introduce external information, context, or general knowledge into your response.\n" 

            "- The dataset can be used to asses specific values of features for companies.\n" 

            "- Make sure the response is user-friendly and concise.\n\n" 
        )
    )

    llm_response = ChatOpenAI(model="gpt-4", temperature=0.1, openai_api_key=openai_api_key)

    # Create a runnable sequence combining the prompt and the LLM
    response_chain = response_prompt | llm_response

    # Use the `invoke` method instead of `run`
    response = response_chain.invoke({
        'shap_values_before': shap_table_before,
        'shap_values_after': shap_table_after,
        'user_query': user_query,
        'filter_representation': final_filter_representation,
        'final_dataset_before': final_dataset_before,
        'final_dataset_after': final_dataset_after,
        'importance_table_before': importance_table_before,
        'importance_table_after': importance_table_after,
        'summary_stats_after': summary_stats_after,
    })

    return response.content