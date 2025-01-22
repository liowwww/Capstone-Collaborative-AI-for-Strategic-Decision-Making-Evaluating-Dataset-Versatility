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

from requirement_2_funcs import get_dataframe_schema, query_to_filter, filter_to_sql, pipeline_extract, calc_SHAP, pred_and_stats
from requirement_3_funcs import get_importance_table, final_importance, ml_to_natural_language


# Global variables to store the uploaded model, dataset, query, label name, and the conversation
uploaded_model = None
uploaded_dataset = None
user_query = ""
label_name = ""
sample_ID = ""
conversation_history = []  # Stores the conversation



# Load the ML model from a .pkl file
def load_model(file):
    """
    Load a machine learning model from a .pkl file.

    Args:
        file (File-like object): The file containing the pickled model.

    Returns:
        str: A message indicating whether the model was loaded successfully or an error occurred.
    """

    global uploaded_model
    try:
        uploaded_model = joblib.load(file.name)
        return "Model loaded successfully."
    except Exception as e:
        return f"Failed to load model: {e}"



# Load the dataset from a CSV file
def load_dataset(file):
    """
    Load a dataset from a CSV file.

    Args:
        file (File-like object): The file containing the dataset in CSV format.

    Returns:
        str: A message indicating whether the dataset was loaded successfully or an error occurred,
             including the shape of the loaded dataset if successful.
    """
    
    global uploaded_dataset
    try:
        uploaded_dataset = pd.read_csv(file.name)
        return f"Dataset loaded successfully with {uploaded_dataset.shape[0]} rows and {uploaded_dataset.shape[1]} columns."
    except Exception as e:
        return f"Failed to load dataset: {e}"



# Validate the format of the API key
def validate_api_key(api_key):
    """
    Validate the format of an OpenAI API key.

    Args:
        api_key (str): The API key to validate.

    Returns:
        str: A message indicating whether the API key is valid or invalid.
    """
    # Match a key that starts with 'sk-proj-' followed by a mix of letters, digits, underscores, and hyphens
    if re.match(r"^sk-proj-[A-Za-z0-9_-]+$", api_key):  
        openai.api_key = api_key  # Set the API key if valid
        return "API Key is valid."
    else:
        return "Invalid API key format. Please enter a valid OpenAI API key."



# Store user query and label, then generate a response based on the team's code
def store_query_and_label(query, label, ID):
    """
    Store a user query, label, and sample ID, then generate a response based on the loaded model and dataset.

    Args:
        query (str): The user's query in natural language.
        label (str): The label column name to analyze.
        ID (str): The sample ID for specific data retrieval.

    Returns:
        str: A formatted conversation history in HTML, including the user's query and the chatbot's response.
    """

    global user_query, label_name, conversation_history, sample_ID
    user_query = query
    label_name = label
    sample_ID = ID
    conversation_history.append({"user": user_query, "bot": "Processing response..."})  # Placeholder for the response

    # Assuming your team has logic to generate a natural language response based on these variables:    # extracting the dataset name
    dataset_name = 'the_dataset'

    # getting datacontext
    schema_with_examples = get_dataframe_schema(uploaded_dataset, dataset_name)

    # extracting model from pipeline and generating code, this needs to be run only once at the beginning
    transformations, model = pipeline_extract(uploaded_model)

    # getting filter representation
    filter_representation = query_to_filter(openai.api_key, schema_with_examples, user_query)

    # getting sql code
    update_query, retrieve_query = filter_to_sql(openai.api_key, schema_with_examples, filter_representation)

    # getting SHAP values and relevant datasets
    before_dataset, shap_values_before, after_dataset, shap_values_after = calc_SHAP(transformations, model, uploaded_dataset, dataset_name, update_query, retrieve_query, label_name, sample_ID)

    # getting final predictions and aggregate statistics
    final_dataset_before, summary_stats_before, final_dataset_after, summary_stats_after = pred_and_stats(uploaded_model, before_dataset, after_dataset, label_name, sample_ID)

    # getting relevant importance tables
    importance_table_before, importance_table_after, feature_names = final_importance(shap_values_before, shap_values_after, final_dataset_after, label_name, sample_ID)

    # getting the final response
    response = ml_to_natural_language(shap_values_before, shap_values_after, feature_names, user_query, filter_representation, final_dataset_before, final_dataset_after, importance_table_before, importance_table_after, summary_stats_after, openai.api_key, label_name, sample_ID)
    conversation_history[-1]["bot"] = response  # Update the last response

    # Display the conversation history with bubble styles
    conversation_display = ""
    for message in conversation_history:
        conversation_display += f"<div class='user-bubble'>User: {message['user']}</div>"
        conversation_display += f"<div class='bot-bubble'>Chatbot: {message['bot']}</div>"

    return conversation_display

# Create Gradio interface
def create_interface():
    with gr.Blocks(css="""
        #header {
            font-size: 32px !important;
            font-weight: bold !important;
            text-align: center !important;
            margin-bottom: 20px !important;
        }
        #description {
            font-size: 18px !important;
            margin-bottom: 20px !important;
            text-align: center !important;
        }
        .upload-box {
            padding: 20px !important;
            border: 1px solid #ccc !important;
            border-radius: 10px !important;
            margin-bottom: 20px !important;
            background-color: #f7f7f7 !important;
        }
        .status-box {
            padding: 20px !important;
            border: 1px solid #ccc !important;
            border-radius: 10px !important;
            margin-bottom: 20px !important;
            background-color: #f7f7f7 !important;
        }
        .chat-box {
            font-size: 20px !important;
            padding: 10px !important;
            background-color: #000 !important;  /* Set chatbox background to black */
            color: #fff !important;  /* Set text color to white */
            border: 1px solid #ccc !important;  /* Ensure all borders match */
            border-radius: 10px !important;
            overflow: hidden !important; /* Remove the scrolling option */
            min-height: 300px !important;  /* Set minimum height for the chatbox */
            overflow-y: auto !important;  /* Allow scrolling only if necessary */
        }
        .query-section {
            font-size: 20px !important;
        }
        .submit-button {
            font-size: 18px !important;
            padding: 10px 20px !important;
        }
        .user-bubble {
            background-color: #333333 !important;  /* Dark gray for user */
            padding: 10px !important;
            border-radius: 15px !important;
            margin: 5px 0 !important;
            max-width: 75% !important;
            word-wrap: break-word !important;
            color: white !important;  /* Set text color inside user bubble to white */
        }
        .bot-bubble {
            background-color: #d3d3d3 !important;  /* Light gray for chatbot */
            padding: 10px !important;
            border-radius: 15px !important;
            margin: 5px 0 !important;
            max-width: 75% !important;
            word-wrap: break-word !important;
            color: black !important;  /* Set text color inside bot bubble to black */
        }
    """) as demo:
        gr.Markdown("# Chatbot-like Interface with ML Model Support", elem_id="header")
        gr.Markdown("Upload a machine learning model (.pkl) and a dataset (.csv). Then, engage in a conversation!", elem_id="description")

        # Upload section in its own box
        with gr.Row():
            with gr.Column():
                with gr.Column():
                    model_upload = gr.File(label="Upload ML Model (.pkl)", file_types=[".pkl"])
                with gr.Column():
                    dataset_upload = gr.File(label="Upload Dataset (.csv)", file_types=[".csv"])
        api_key_input = gr.Textbox(label="Enter ChatGPT API Key", placeholder="Enter your OpenAI API key here", type="password")
        label_input = gr.Textbox(label="Label Name", placeholder="Type the label name here...", lines=1)  # Label Name input box
        sample_ID = gr.Textbox(label="Sample ID", placeholder="Type the name of the column containing sample IDs here...", lines=1)  # Sample ID input box


        # Status section in its own box
        with gr.Row():
            model_status = gr.Textbox(label="Model Status", placeholder="Model status will appear here...", interactive=False)
            dataset_status = gr.Textbox(label="Dataset Status", placeholder="Dataset status will appear here...", interactive=False)
            api_key_status = gr.Textbox(label="API Key Status", placeholder="API key status will appear here...", interactive=False)

        # Chatbox section for conversation
        with gr.Column():
            chat_box = gr.HTML(elem_classes=["chat-box"])
            user_input = gr.Textbox(label="Your Message", placeholder="Type your message here...", lines=2, elem_classes=["query-section"])
            submit_btn = gr.Button("Send Message", elem_classes=["submit-button"])

        # Actions
        model_upload.change(load_model, inputs=model_upload, outputs=model_status)
        dataset_upload.change(load_dataset, inputs=dataset_upload, outputs=dataset_status)
        api_key_input.change(validate_api_key, inputs=api_key_input, outputs=api_key_status)  # Validate the API key
        submit_btn.click(store_query_and_label, inputs=[user_input, label_input, sample_ID], outputs=chat_box)


    return demo

# Launch the Gradio interface
interface = create_interface()
interface.launch()