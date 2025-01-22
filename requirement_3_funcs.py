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
    Calculates the mean absolute SHAP values for each feature and returns a sorted DataFrame of feature importance.

    Args:
        shap_values (np.ndarray): A numpy array of SHAP values (n_samples x n_features).
        feature_names (list or np.ndarray): A list or array of feature names corresponding to the SHAP values.

    Returns:
        pd.DataFrame: A DataFrame containing feature names and their corresponding importance, sorted in descending order.
    """
    # Validate inputs
    if not isinstance(shap_values, np.ndarray):
        raise ValueError("shap_values must be a numpy array.")
    if not isinstance(feature_names, (list, np.ndarray)):
        raise ValueError("feature_names must be a list or numpy array.")
    if shap_values.shape[1] != len(feature_names):
        raise ValueError("The number of features in shap_values must match the length of feature_names.")
    
    # Calculate the mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create a DataFrame to store the feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    })

    # Sort by importance in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    return importance_df



def final_importance(shap_values_before, shap_values_after, final_dataset_after, label_name, sample_ID):
    """
    Generates importance tables for SHAP values before and after an update.

    Args:
        shap_values_before (np.ndarray or None): SHAP values before the update, or None if no "before" dataset exists.
        shap_values_after (np.ndarray): SHAP values after the update.
        final_dataset_after (pd.DataFrame): Dataset after updates, including predictions.
        label_name (str): Name of the column containing labels.
        sample_ID (str): Name of the column containing row identifiers.

    Returns:
        tuple: Contains the following:
            - importance_table_before (pd.DataFrame or None): Feature importance table before the update (or None).
            - importance_table_after (pd.DataFrame): Feature importance table after the update.
            - feature_names (list): List of feature names used in the importance tables.
    """
    # Validate inputs
    if not isinstance(final_dataset_after, pd.DataFrame):
        raise ValueError("final_dataset_after must be a pandas DataFrame.")
    if not isinstance(label_name, str) or not isinstance(sample_ID, str):
        raise ValueError("label_name and sample_ID must be strings.")
    if shap_values_before is not None and not isinstance(shap_values_before, np.ndarray):
        raise ValueError("shap_values_before must be a numpy array or None.")
    if not isinstance(shap_values_after, np.ndarray):
        raise ValueError("shap_values_after must be a numpy array.")

    # Extract feature names
    try:
        shap_frame = final_dataset_after.drop(columns=[sample_ID, 'prediction', label_name])
    except KeyError as e:
        raise KeyError(f"Missing expected column in final_dataset_after: {e}")

    feature_names = list(shap_frame.columns)

    # Generate importance tables
    if shap_values_before is None:
        importance_table_before = None
    else:
        importance_table_before = get_importance_table(shap_values_before, feature_names)

    importance_table_after = get_importance_table(shap_values_after, feature_names)

    return importance_table_before, importance_table_after, feature_names
    

def ml_to_natural_language(
    shap_values_before, shap_values_after, feature_names, user_query, filter_representation,
    final_dataset_before, final_dataset_after, importance_table_before, importance_table_after,
    summary_stats_after, openai_api_key, label_name, sample_ID
):
    """
    Converts machine learning model outputs into natural language responses based on input datasets and SHAP values.

    Args:
        shap_values_before (np.ndarray): SHAP values before dataset updates.
        shap_values_after (np.ndarray): SHAP values after dataset updates.
        feature_names (list): List of feature names corresponding to SHAP values.
        user_query (str): Natural language query from the user.
        filter_representation (str): Filter representation corresponding to the user's query.
        final_dataset_before (pd.DataFrame or None): Dataset before updates.
        final_dataset_after (pd.DataFrame): Dataset after updates.
        importance_table_before (pd.DataFrame or None): Feature importance table before updates.
        importance_table_after (pd.DataFrame): Feature importance table after updates.
        summary_stats_after (pd.DataFrame): Summary statistics of the dataset after updates.
        openai_api_key (str): API key for authenticating with OpenAI.

    Returns:
        str: Natural language response generated by the model.
    """
    # Validate inputs
    if not isinstance(filter_representation, str):
        raise ValueError("filter_representation must be a string.")
    if not isinstance(user_query, str):
        raise ValueError("user_query must be a string.")
    if not isinstance(feature_names, list):
        raise ValueError("feature_names must be a list.")
    if not isinstance(final_dataset_after, pd.DataFrame):
        raise ValueError("final_dataset_after must be a pandas DataFrame.")
    if shap_values_before is not None and not isinstance(shap_values_before, np.ndarray):
        raise ValueError("shap_values_before must be a numpy array or None.")
    if not isinstance(shap_values_after, np.ndarray):
        raise ValueError("shap_values_after must be a numpy array.")

    # Adjust the filter representation
    final_filter_representation = filter_representation.replace("[E]", "").strip()

    # Create DataFrames for SHAP values
    shap_table_before = pd.DataFrame(shap_values_before, columns=feature_names) if shap_values_before is not None else None
    shap_table_after = pd.DataFrame(shap_values_after, columns=feature_names)

    # Define the prompt template
    response_prompt = PromptTemplate(
        input_variables=[
            "final_filter_representation", "shap_values_before",
            "shap_values_after", "final_dataset_before", "final_dataset_after",
            "importance_table_before", "importance_table_after", "summary_stats_after"
        ],
        template=(
            "You are an assistant designed to help bank employees analyze and explain credit ratings for customers applying for loans. " 
            "Your job is to provide clear, concise, and accurate answers strictly based on the provided data. " 
            "Do not make assumptions or use external knowledge beyond the given inputs.\n\n" 

             

            "### Instructions:\n" 
            "- Provide an answer strictly based on the filter representation: {final_filter_representation}.\n" 
            "- Use the input data to generate insights that align with this representation.\n" 
            "- If the required data is missing or insufficient to answer the query, state: 'The data provided is insufficient to answer this query.'\n" 
            "- Do not provide explanations about the process of finding the answer.\n" 
            "- Avoid introducing external information, context, or general knowledge into your response.\n\n" 


            "### Dataset and Model Explanation:\n" 
            "The dataset contains information about 200 clients of a bank, including various features about their financial history, loan details, and other factors. " 
            "The dataset includes a prediction of their credit risk rating based on the column 'prediction':\n" 
            "- A value of 1 in 'prediction' indicates a good credit risk (low likelihood of default).\n" 
            "- A value of 0 in 'prediction' indicates a bad credit risk (higher likelihood of default).\n\n" 


            "### Relevant Data:\n" 
            "Here is the relevant information from the user query: {final_filter_representation}\n\n" 
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
            "- The dataset can be used to asses specific values of features for individuals.\n" 
            "- Make sure the response is user-friendly and concise.\n\n" 
        )
    )

    # Initialize the language model
    llm_response = ChatOpenAI(model="gpt-4", temperature=0.1, openai_api_key=openai_api_key)

    # Create a runnable sequence combining the prompt and the LLM
    response_chain = response_prompt | llm_response

    # Generate the response
    try:
        response = response_chain.invoke({
            'shap_values_before': shap_table_before,
            'shap_values_after': shap_table_after,
            'final_filter_representation': final_filter_representation,
            'final_dataset_before': final_dataset_before, #.loc[:, [sample_ID, 'prediction', label_name]] if final_dataset_before is not None else None,
            'final_dataset_after': final_dataset_after, #.loc[:, [sample_ID, 'prediction', label_name]],
            'importance_table_before': importance_table_before, #.head(6) if importance_table_before is not None else None,
            'importance_table_after': importance_table_after, #.head(6),
            'summary_stats_after': summary_stats_after.loc[['count', 'mean', 'max', 'min']],
        })
    except Exception as e:
        raise RuntimeError(f"An error occurred while generating the response: {e}")

    return response.content