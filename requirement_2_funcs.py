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


def get_dataframe_schema(dataset, dataset_name):
    """
    Generates a schema representation of a pandas DataFrame including column names, data types, 
    and a few example rows.

    Args:
        dataset (pd.DataFrame): The DataFrame to extract schema information from.
        dataset_name (str): A descriptive name for the dataset.

    Returns:
        str: A formatted string containing the table schema and example rows.
    """
    # Ensure the dataset is a pandas DataFrame
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("The input dataset must be a pandas DataFrame.")
    
    # Create the column schema section
    column_schema = "\n".join(
        [f"- {col} ({dtype})" for col, dtype in zip(dataset.columns, dataset.dtypes)]
    )
    
    # Extract a few example rows
    example_rows = dataset.head(3).to_string(index=False)
    
    # Format the schema with the dataset name and examples
    schema_with_examples = (
        f"Table Name: {dataset_name}\n\n"
        f"Columns:\n{column_schema}\n\n"
        f"Example Rows:\n{example_rows}"
    )
    
    return schema_with_examples



def query_to_filter(openai_api_key, schema_with_examples, user_query):
    """
    Converts a natural language user query into a filter representation using OpenAI's language model.

    Args:
        openai_api_key: The API key for authenticating with OpenAI.
        schema_with_examples: A formatted string containing dataset schema and example rows.
        user_query: The user's natural language query to be converted.

    Returns:
        str: The filter representation generated from the user query.
    """
    # Path to the examples file
    filter_examples_path = r'Prompt_Examples\User_Query_to_Filter_Repr_Eg.txt'
    
    # Read filter examples from the file
    try:
        with open(filter_examples_path, 'r') as file:
            filter_examples = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Examples file not found at {filter_examples_path}. Please verify the path."
        )
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the examples file: {e}")

    # Define the prompt template
    template_filter_prompt = '''
    You are a highly capable assistant that translates natural language queries into filter representations based on the given dataset context.

    ### Dataset Context
    {data_context}

    ### Examples
    {filter_examples}

    ### Task
    Translate the following natural language query into a filter representation.

    ### Input
    Natural Language Query: {natural_language_query}

    ### Output
    - Output only the filter representation as a single string. Do not include any additional explanation or comments.
    - Set True and False to 1 and 0 in the Filter Representation if the dataset feature values are only numerical.
    '''

    # Construct the prompt
    filter_prompt = PromptTemplate(
        input_variables=['natural_language_query', 'data_context', 'filter_examples'],
        template=template_filter_prompt
    )

    # Initialize the language model
    llm_filter = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)

    # Combine the prompt template and the language model into a sequence
    filter_chain = filter_prompt | llm_filter

    # Generate the filter representation
    try:
        filter_representation = filter_chain.invoke({
            'natural_language_query': user_query,
            'data_context': schema_with_examples,
            'filter_examples': filter_examples
        })
    except Exception as e:
        raise RuntimeError(f"An error occurred while generating the filter representation: {e}")

    return filter_representation.content



def filter_to_sql(openai_api_key, schema_with_examples, filter_representation):
    """
    Converts a filter representation into corresponding SQL code using OpenAI's language model.

    Args:
        openai_api_key: The API key for authenticating with OpenAI.
        schema_with_examples: A formatted string containing dataset schema and example rows.
        filter_representation: The filter representation to convert into SQL code.

    Returns:
        tuple: A tuple containing the update SQL query and the retrieval SQL query.
    """
    # Path to the examples file
    sql_examples_path = r'Prompt_Examples\Filter_Repr_to_SQL_Eg.txt'
    
    # Read SQL examples from the file
    try:
        with open(sql_examples_path, 'r') as file:
            sql_examples = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Examples file not found at {sql_examples_path}. Please verify the path."
        )
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the examples file: {e}")

    # Define the prompt template
    template_sql_prompt = '''
    You are a highly capable assistant that translates filter representation queries into SQL code based on the given dataset context. 

    ### Dataset Context
    {data_context}

    ### Examples
    {sql_examples}

    ### Task
    Translate the following filter representation query into SQL code.

    ### Input
    Filter Representation: {filter_representation}

    ### Output
    - First, generate the SQL code for the filter or update operation all on ONE LINE.
    - ON A NEW LINE, provide the SQL code for the retrieval operation.
    - Only use SELECT or UPDATE methods in the SQL code.
    - Set True and False to 1 and 0 in the SQL Code if the dataset feature values are only numerical.
    - If the filter representation asks for top features, then select all the data.
    - Always write column names in double quotations.

    Output only the SQL code in the following format:
    <Filter/Update SQL Code> <Retrieval SQL Code>
    '''

    # Construct the prompt
    sql_prompt = PromptTemplate(
        input_variables=['filter_representation', 'data_context', 'sql_examples'],
        template=template_sql_prompt
    )

    # Initialize the language model
    llm_sql = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)

    # Combine the prompt and the language model into a sequence
    sql_chain = sql_prompt | llm_sql

    # Generate SQL code
    try:
        sql_queries = sql_chain.invoke({
            'filter_representation': filter_representation,
            'data_context': schema_with_examples,
            'sql_examples': sql_examples
        })
    except Exception as e:
        raise RuntimeError(f"An error occurred while generating the SQL queries: {e}")

    # Parse the result to split update and retrieve code
    sql_queries = sql_queries.content.strip().split("\n")
    update_query = sql_queries[0].strip()
    retrieve_query = sql_queries[1].strip() if len(sql_queries) > 1 else update_query

    return update_query, retrieve_query



def filter_to_sql_validation(openai_api_key, schema_with_examples, filter_representation):
    """
    Converts a filter representation into corresponding SQL code using OpenAI's language model. Modified for validation by not splitting at newline character

    Args:
        openai_api_key: The API key for authenticating with OpenAI.
        schema_with_examples: A formatted string containing dataset schema and example rows.
        filter_representation: The filter representation to convert into SQL code.

    Returns:
        string: A string containing the update SQL query and the retrieval SQL query.
    """
    # Path to the examples file
    sql_examples_path = r'Prompt_Examples\Filter_Repr_to_SQL_Eg.txt'
    
    # Read SQL examples from the file
    try:
        with open(sql_examples_path, 'r') as file:
            sql_examples = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Examples file not found at {sql_examples_path}. Please verify the path."
        )
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the examples file: {e}")

    # Define the prompt template
    template_sql_prompt = '''
    You are a highly capable assistant that translates filter representation queries into SQL code based on the given dataset context. 

    ### Dataset Context
    {data_context}

    ### Examples
    {sql_examples}

    ### Task
    Translate the following filter representation query into SQL code.

    ### Input
    Filter Representation: {filter_representation}

    ### Output
    - First, generate the SQL code for the filter or update operation all on ONE LINE.
    - ON A NEW LINE, provide the SQL code for the retrieval operation.
    - Only use SELECT or UPDATE methods in the SQL code.
    - Set True and False to 1 and 0 in the SQL Code if the dataset feature values are only numerical.
    - If the filter representation asks for top features, then select all the data.
    - Always write column names in double quotations.

    Output only the SQL code in the following format:
    <Filter/Update SQL Code> <Retrieval SQL Code>
    '''

    # Construct the prompt
    sql_prompt = PromptTemplate(
        input_variables=['filter_representation', 'data_context', 'sql_examples'],
        template=template_sql_prompt
    )

    # Initialize the language model
    llm_sql = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)

    # Combine the prompt and the language model into a sequence
    sql_chain = sql_prompt | llm_sql

    # Generate SQL code
    try:
        sql_queries = sql_chain.invoke({
            'filter_representation': filter_representation,
            'data_context': schema_with_examples,
            'sql_examples': sql_examples
        })
    except Exception as e:
        raise RuntimeError(f"An error occurred while generating the SQL queries: {e}")


    return sql_queries.content



from sklearn.preprocessing import FunctionTransformer

def pipeline_extract(pipeline):
    """
    Extracts the transformations and model from a scikit-learn pipeline.

    Args:
        pipeline: A scikit-learn pipeline object.

    Returns:
        tuple: A tuple containing two elements:
            - transformations: transformation steps from the pipeline (all steps except the last one).
            - model: The final model step from the pipeline.
    """
    # Validate input is a scikit-learn pipeline
    if not hasattr(pipeline, "steps"):
        raise ValueError("The input must be a scikit-learn pipeline object with a 'steps' attribute.")

    # Check if the pipeline has more than one step
    if len(pipeline.steps) > 1:
        # Extract transformations (all but the last step) and the final model (last step)
        transformations = pipeline[:-1]
        model = pipeline[-1]
    else:
        # Pipeline consists only of the model, use a "do-nothing" transformation
        transformations = Pipeline([('noop', FunctionTransformer(lambda x: x))])  # Identity transformer
        model = pipeline.steps[0][1]  # Extract the model

    return transformations, model



def calc_SHAP(transformations, model, dataset, dataset_name, update_query, retrieve_query, label_name, sample_ID):
    """
    Calculates SHAP values for a dataset before and after applying an update query.

    Args:
        transformations: Transformation pipeline for preprocessing the dataset.
        model: A trained scikit-learn model.
        dataset: A pandas DataFrame containing the dataset.
        dataset_name: Name of the table in the in-memory SQLite database.
        update_query: SQL query to update the dataset.
        retrieve_query: SQL query to retrieve the dataset after the update.
        label_name: Name of the column containing labels.
        sample_ID: Name of the column containing row identifiers.

    Returns:
        tuple: Contains the following:
            - before_dataset (pd.DataFrame): Dataset before applying the update query (None if no 'UPDATE').
            - shap_values_before: SHAP values for the dataset before the update (None if no 'UPDATE').
            - after_dataset (pd.DataFrame): Dataset after applying the update query.
            - shap_values_after: SHAP values for the dataset after the update.
    """
    def initialize_database(dataset, dataset_name):
        connection = sqlite3.connect(':memory:')
        dataset.to_sql(dataset_name, connection, if_exists='replace', index=False)
        return connection

    def get_dataset_from_query(connection, query):
        return pd.read_sql(query, connection)

    def initialize_explainer(model, data_transformed):
        # Check for TreeExplainer
        if isinstance(model, (sklearn.ensemble.RandomForestClassifier, sklearn.ensemble.GradientBoostingClassifier)):
            return shap.TreeExplainer(model, check_additivity=False)  # Avoid additivity check for tree-based models
        # Check for LogisticRegression
        elif isinstance(model, sklearn.linear_model.LogisticRegression):
            return shap.LinearExplainer(model, data_transformed)  # Additivity defaults to True for LinearExplainer
        # Check for LinearRegression
        elif isinstance(model, sklearn.linear_model.LinearRegression):
            return shap.LinearExplainer(model, data_transformed)  # Additivity defaults to True for LinearExplainer
        # Default to KernelExplainer
        else:
            return shap.KernelExplainer(model.predict_proba, data_transformed)  # KernelExplainer does not check additivity
        
    def calculate_shap_values(explainer, data_transformed):
        return explainer.shap_values(data_transformed)

    try:
        # Create SQLite database and retrieve initial dataset
        connection = initialize_database(dataset, dataset_name)

        if 'UPDATE' in update_query:
            # Dataset and SHAP values before update
            before_dataset = get_dataset_from_query(connection, retrieve_query)
            sql_dataset = before_dataset.drop(columns=[sample_ID, label_name])
            sql_dataset_transformed = transformations.fit_transform(sql_dataset)
            explainer = initialize_explainer(model, sql_dataset_transformed)
            shap_values_before = calculate_shap_values(explainer, sql_dataset_transformed)
            
            # Apply update query and retrieve updated dataset
            cursor = connection.cursor()
            cursor.execute(update_query)
            connection.commit()

            after_dataset = get_dataset_from_query(connection, retrieve_query)
            sql_dataset = after_dataset.drop(columns=[sample_ID, label_name])
            sql_dataset_transformed = transformations.fit_transform(sql_dataset)
            explainer = initialize_explainer(model, sql_dataset_transformed)
            shap_values_after = calculate_shap_values(explainer, sql_dataset_transformed)

        else:
            before_dataset = None
            shap_values_before = None

            # Apply update query and retrieve updated dataset
            cursor = connection.cursor()
            cursor.execute(update_query)
            connection.commit()

            after_dataset = get_dataset_from_query(connection, update_query)
            sql_dataset = after_dataset.drop(columns=[sample_ID, label_name])
            sql_dataset_transformed = transformations.fit_transform(sql_dataset)
            explainer = initialize_explainer(model, sql_dataset_transformed)
            shap_values_after = calculate_shap_values(explainer, sql_dataset_transformed)


    except Exception as e:
        raise RuntimeError(f"An error occurred during SHAP value calculation: {e}")

    finally:
        connection.close()

    return before_dataset, shap_values_before, after_dataset, shap_values_after
        


def pred_and_stats(pipeline, before_dataset, after_dataset, label_name, sample_ID):
    """
    Generates predictions using a scikit-learn pipeline and computes aggregate statistics
    for datasets before and after updates.

    Args:
        pipeline: A scikit-learn pipeline object with a model as the final step.
        before_dataset: DataFrame before updates (or None if no updates occurred).
        after_dataset: DataFrame after updates.
        label_name: Name of the column containing labels.
        sample_ID: Name of the column containing row identifiers.

    Returns:
        tuple: Contains the following:
            - final_dataset_before (pd.DataFrame or None): Dataset before updates with predictions appended.
            - summary_stats_before (pd.DataFrame or None): Descriptive statistics for the dataset before updates.
            - final_dataset_after (pd.DataFrame): Dataset after updates with predictions appended.
            - summary_stats_after (pd.DataFrame): Descriptive statistics for the dataset after updates.
    """
    def append_predictions(dataset, pipeline, label_name, sample_ID):
        """Helper function to generate predictions and append them to the dataset."""
        features = dataset.drop(columns=[sample_ID, label_name])
        predictions = pipeline.predict(features)
        predictions_df = pd.DataFrame(predictions, columns=['prediction'])
        return pd.concat([dataset, predictions_df], axis=1)

    def compute_summary_stats(dataset):
        """Helper function to compute descriptive statistics."""
        return dataset.describe()

    if before_dataset is None:
        # Handle the case where no "before" dataset exists
        final_dataset_before = None
        summary_stats_before = None

        final_dataset_after = append_predictions(after_dataset, pipeline, label_name, sample_ID)
        summary_stats_after = compute_summary_stats(final_dataset_after)
    else:
        # Process both "before" and "after" datasets
        final_dataset_before = append_predictions(before_dataset, pipeline, label_name, sample_ID)
        summary_stats_before = compute_summary_stats(final_dataset_before)

        final_dataset_after = append_predictions(after_dataset, pipeline, label_name, sample_ID)
        summary_stats_after = compute_summary_stats(final_dataset_after)

    return final_dataset_before, summary_stats_before, final_dataset_after, summary_stats_after