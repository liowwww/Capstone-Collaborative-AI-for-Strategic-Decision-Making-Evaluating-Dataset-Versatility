import openai
import pandas as pd
import joblib
import gradio as gr

# Global variables to store the uploaded model, dataset, query, label, response, and final response
uploaded_model = None
uploaded_dataset = None
user_query = ""
label_name = ""
response_output = "This is a placeholder response that will be updated later."
final_response = ""  # Initializing final_response as an empty string


# Load the ML model from a .pkl file
def load_model(file):
    global uploaded_model
    try:
        uploaded_model = joblib.load(file.name)
        return "Model loaded successfully."
    except Exception as e:
        return f"Failed to load model: {e}"


# Load the dataset from a CSV file
def load_dataset(file):
    global uploaded_dataset
    try:
        uploaded_dataset = pd.read_csv(file.name)
        return f"Dataset loaded successfully with {uploaded_dataset.shape[0]} rows and {uploaded_dataset.shape[1]} columns."
    except Exception as e:
        return f"Failed to load dataset: {e}"


# Store user query and label, and set the final response (this can be updated with actual logic later)
def store_query_and_label(query, label):
    global user_query, label_name, final_response
    user_query = query
    label_name = label
    final_response = ""  # Reset final response when query and label are stored
    return final_response  # Return the empty final_response (for now)


# Create Gradio interface
def create_interface():
    with gr.Blocks(css="""
        #header {
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        #description {
            font-size: 18px;
            margin-bottom: 20px;
            text-align: center;
        }
        .query-section, .response-section {
            font-size: 20px;
        }
        .query-section label, .response-section label {
            font-size: 20px !important;
            font-weight: bold;
        }
        .submit-button {
            font-size: 18px;
            padding: 10px 20px;
        }
    """) as demo:
        gr.Markdown("# Conversational Interface with ML Model Support", elem_id="header")
        gr.Markdown("Upload a machine learning model (.pkl) and a dataset (.csv). Then, type a query and store it.", elem_id="description")

        # Upload section
        with gr.Row():
            model_upload = gr.File(label="Upload ML Model (.pkl)", file_types=[".pkl"])
            dataset_upload = gr.File(label="Upload Dataset (.csv)", file_types=[".csv"])
        model_status = gr.Textbox(label="Model Status", placeholder="Model status will appear here...", interactive=False)
        dataset_status = gr.Textbox(label="Dataset Status", placeholder="Dataset status will appear here...", interactive=False)

        # Query, Label and Response section
        with gr.Column():
            user_input = gr.Textbox(label="Your Query", placeholder="Type your query here...", lines=2, elem_classes=["query-section"])
            label_input = gr.Textbox(label="Label Name", placeholder="Type the label name here...", lines=1)  # New Label text box
            output = gr.Textbox(label="Response", placeholder="Response will appear here...", lines=5, interactive=False, elem_classes=["response-section"])
            submit_btn = gr.Button("Submit Query", elem_classes=["submit-button"])

        # Actions
        model_upload.change(load_model, inputs=model_upload, outputs=model_status)
        dataset_upload.change(load_dataset, inputs=dataset_upload, outputs=dataset_status)
        submit_btn.click(store_query_and_label, inputs=[user_input, label_input], outputs=output)

    return demo


# Launch the Gradio interface
interface = create_interface()
interface.launch()

print("User Query:", user_query)
print("Label Name:", label_name)
print("Uploaded Dataset:")
print(uploaded_dataset)
print("Uploaded Model (Serialized):", uploaded_model)