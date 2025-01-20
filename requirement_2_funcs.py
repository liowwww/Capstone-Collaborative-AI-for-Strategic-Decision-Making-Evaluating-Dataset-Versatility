# imports
import numpy as np
import pandas as pd
import pickle as pkl
import os
import openai
import sqlite3
import shap
import sklearn

from sklearn.pipeline import Pipeline
from pandasql import sqldf
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from os.path import dirname, abspath


# function to get a schema from the dataset
def get_dataframe_schema(dataset, dataset_name):

    schema = f"Table: {dataset.columns.name or 'DataFrame'}\nColumns:\n"
    schema += "\n".join([f"- {col} ({dtype})" for col, dtype in zip(dataset.columns, dataset.dtypes)])
    example_rows = dataset.head(3).to_string(index=False)
    data_context = f"Table Name: {dataset_name}\n\n{schema}\n\nExample Rows:\n{example_rows}"
    return data_context


# function for natural language to filter representation 
def query_to_filter(openai_api_key, data_context, user_query):

    filter_examples_path = r'Prompt_Examples\User_Query_to_Filter_Repr_Eg.txt'
    with open(filter_examples_path, 'r') as file:
        filter_examples = file.read()

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

    filter_prompt = PromptTemplate(
        input_variables=['natural_language_query', 'data_context', 'filter_examples'],
        template=template_filter_prompt
    )

    # Initialize the LLM with the updated import
    llm_filter = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)

    # Create a runnable sequence combining the prompt and the LLM
    filter_chain = filter_prompt | llm_filter

    # Use the `invoke` method instead of `run`
    filter_representation = filter_chain.invoke({
        'natural_language_query': user_query,
        'data_context': data_context,
        'filter_examples': filter_examples
    })

    print('---------------------------------')
    print(filter_representation.content)
    return filter_representation.content


# function for filter representation to sql code
def filter_to_sql(openai_api_key, data_context, filter_representation):

    sql_examples_path = r'Prompt_Examples\Filter_Repr_to_SQL_Eg.txt'
    with open(sql_examples_path, 'r') as file:
        sql_examples = file.read()

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
    - First, generate the SQL code for the filter or update operation all one ONE LINE.
    - ON A NEW LINE, provide the SQL code for the retrieval operation.
    - Only used SELECT or UPDATE methods in the SQL code
    - Set True and False to 1 and 0 in the SQL Code if the dataset feature values are only numerical.
    - If filter representation is asking for top features then select all the data
    - Always write column names in double quotations

    Output only the SQL code in the following format:
    <Filter/Update SQL Code> <Retrieval SQL Code>
    '''
        
    sql_prompt = PromptTemplate(
        input_variables=['filter_representation', 'data_context', 'sql_examples'],
        template=template_sql_prompt
    )

    llm_sql = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)

    # Create a runnable sequence combining the prompt and the LLM
    sql_chain = sql_prompt | llm_sql

    # Use the `invoke` method instead of `run`
    sql_queries = sql_chain.invoke({
        'filter_representation': filter_representation,
        'data_context': data_context,
        'sql_examples': sql_examples
    })

    # parsing the result to split update and retrieve code
    sql_queries = sql_queries.content
    queries = sql_queries.strip().split("\n")  # Split by newline character:
    if len(queries) > 1:
        update_query = queries[0].strip()
        retrieve_query = queries[1].strip()
    else:
        update_query = queries[0].strip()
        retrieve_query = queries[0].strip()

    print('---------------------------------')
    print(update_query)
    print(retrieve_query)
    return update_query, retrieve_query


# function for pipeline extraction and code generation
def pipeline_extract(pipeline):

    # extracting the transformations
    transformations = pipeline[:-1]

    # extracting the model
    model = pipeline[-1]

    return transformations, model


# function to get SHAP values and relevant datasets
def calc_SHAP(transformations, model, dataset, dataset_name, update_query, retrieve_query):

    # extracting model and SHAP variables
    if 'UPDATE' in update_query:
        # first get SHAP values for unedited dataset
        # load the dataframe into the database
        # create an in-memory SQLite database
        connection = sqlite3.connect(':memory:')
        # create cursor
        cursor = connection.cursor()
        dataset.to_sql(dataset_name, connection, if_exists='replace', index=False)
        cursor.execute(retrieve_query)
        connection.commit()
        before_dataset = pd.read_sql(retrieve_query, connection)
        sql_dataset = before_dataset.iloc[:, 1:-1]  # HARD CODED
        sql_dataset_transformed = transformations.fit_transform(sql_dataset)

        # extracting an explainer
        if isinstance(model, (sklearn.ensemble.RandomForestClassifier, sklearn.ensemble.GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, sklearn.ensemble.LogisticRegression):
            explainer = shap.LinearExplainer(model, sql_dataset_transformed)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, sql_dataset_transformed)
        
        # importants outputs are edited_dataset and explainer
        shap_values_before = explainer.shap_values(sql_dataset_transformed, check_additivity=False) # SET TO FALSE FOR NOW

        # second get SHAP values for updated dataset
        # load the dataframe into the database
        # create an in-memory SQLite database
        connection = sqlite3.connect(':memory:')
        # create cursor
        cursor = connection.cursor()
        dataset.to_sql(dataset_name, connection, if_exists='replace', index=False)
        cursor.execute(update_query)
        connection.commit()
        after_dataset = pd.read_sql(retrieve_query, connection)
        sql_dataset = after_dataset.iloc[:, 1:-1]  # HARD CODED
        sql_dataset_transformed = transformations.fit_transform(sql_dataset)

        # extracting an explainer
        if isinstance(model, (sklearn.ensemble.RandomForestClassifier, sklearn.ensemble.GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, sklearn.ensemble.LogisticRegression):
            explainer = shap.LinearExplainer(model, sql_dataset_transformed)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, sql_dataset_transformed)
        
        # importants outputs are edited_dataset and explainer
        shap_values_before = explainer.shap_values(sql_dataset_transformed, check_additivity=False) # SET TO FALSE FOR NOW
        
        # importants outputs are edited_dataset and explainer
        shap_values_after = explainer.shap_values(sql_dataset_transformed, check_additivity=False) # SET TO FALSE FOR NOW

        return before_dataset, shap_values_before, after_dataset, shap_values_after
        
    else:
        # load the dataframe into the database
        # create an in-memory SQLite database
        connection = sqlite3.connect(':memory:')
        # create cursor
        cursor = connection.cursor()
        dataset.to_sql(dataset_name, connection, if_exists='replace', index=False)
        cursor.execute(update_query)
        connection.commit()
        after_dataset = pd.read_sql(update_query, connection)
        sql_dataset = after_dataset.iloc[:, 1:-1]  # HARD CODED
        sql_dataset_transformed = transformations.fit_transform(sql_dataset)

        # extracting an explainer
        if isinstance(model, (sklearn.ensemble.RandomForestClassifier, sklearn.ensemble.GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, sklearn.ensemble.LogisticRegression):
            explainer = shap.LinearExplainer(model, sql_dataset_transformed)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, sql_dataset_transformed)
        
        # importants outputs are edited_dataset and explainer
        shap_values_before = explainer.shap_values(sql_dataset_transformed, check_additivity=False) # SET TO FALSE FOR NOW

        # importants outputs are edited_dataset and explainer
        shap_values_after = explainer.shap_values(sql_dataset_transformed, check_additivity=False) # SET TO FALSE FOR NOW
        before_dataset = None
        shap_values_before = None

        return before_dataset, shap_values_before, after_dataset, shap_values_after


# function for making final prediction and getting aggregate statistics
def pred_and_stats(pipeline, before_dataset, after_dataset):

    if before_dataset is None:
        # making and appending predictions if no values were updated
        predictions_after = pipeline.predict(after_dataset.iloc[:, 1:-1]) # HARD CODED
        predictions_after_df = pd.DataFrame(predictions_after, columns=['prediction'])
        final_dataset_before = None
        final_dataset_after = pd.concat([after_dataset, predictions_after_df], axis=1)

        summary_stats_before = None
        summary_stats_after = final_dataset_after.describe()

        return final_dataset_before, summary_stats_before, final_dataset_after, summary_stats_after


    else:
        # making and appending predictions before values were updated
        predictions_before = pipeline.predict(before_dataset.iloc[:, 1:-1]) # HARD CODED
        predictions_before_df = pd.DataFrame(predictions_before, columns=['prediction'])
        final_dataset_before = pd.concat([before_dataset, predictions_before_df], axis=1)

        # making and appending predictions after values were updated
        predictions_after = pipeline.predict(after_dataset.iloc[:, 1:-1]) # HARD CODED
        predictions_after_df = pd.DataFrame(predictions_after, columns=['prediction'])
        final_dataset_after = pd.concat([after_dataset, predictions_after_df], axis=1)

        summary_stats_before = final_dataset_before.describe()
        summary_stats_after = final_dataset_after.describe()

        return final_dataset_before, summary_stats_before, final_dataset_after, summary_stats_after
