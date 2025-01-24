The project involves leveraging LLMs through prompt engineering in order to answer user queries regarding a dataset and a sci-kit learn machine learning model. 
If you want to try the system there are two models and datasets, one related to loans and one related to CO2 emissions. 


Installing the environment:
1. Make sure you have Anaconda installed on your system.
2. In the project folder, you’ll find a .yml file in the environments folder:
3. To install the environment, open Anaconda Prompt and type the following, replacing environment.yml with the path to the relevant .yml file:
- conda env create -f environment.yml
4. After the environment is created, activate it by typing the following:
- For the CO2 model: conda activate capstone_d2
5. Install Gradio in the environment by typing the following in Anaconda Prompt:
- pip install gradio
6. Now, open the repository in Visual Studio Code. Make sure to activate the appropriate environment based on the model you are using.


Running the system:
1. Open and run the 'requirement_2_funcs.py' file
2. Open and run the 'requirement_3_funcs.py' file
3. Open and run the 'main.py' file
4. If gradio was installed correctly a link to open the interface should be present in the terminal, click it to open
5. In the 'loan_model_dataset' and 'co2_model_dataset' folders there will be a .csv file and a .pkl file for the dataset and model pipline repsectively
6. On the interface there will be a place to drag and drop the .pkl file and .csv file respectively
7. Then type the openai API key joined in the email - if you already have an API key we suggest you use your own as there are limited tokens which cost additional money - bear this in mind when playing around with the model too
8. If you are using the loan dataset the Label Name is 'y' or if you are using the co2 dataset the Label Name is 'CO2Emissions', this helps the system identify which column is the labels
9. If you are using the loan dataset the Sample ID is 'ID' or if you are using the co2 dataset the Sample ID is 'Country', this helps the system identify which column has the row names
10. Now a question about the dataset can be asked, we would suggest trying the following questions or something similar, keep in mind follow up questions cannot be performed. Rows or columns referred to in the question must be present in the dataset.

Questions Loan Dataset:
- How many people have a loan duration more than 40 weeks?
- Can you describe the data?
- What are the most important features?
- What is the average age of people in the dataset?
- What are important features for the client with id 191?'
- What is your reasoning for deciding if people with no missed payments are good credit risk?
- How likely are people that are older than 30 to be good credit risk?
- If id 631 was unemployed what would be his credit risk?
- What is the average age of clients with loan amounts greater than 8000?
- If people in the data were unemployed, what are important features for predicting credit risk?
- If we were to increase the loan amount by 250, what would happen to the likelihood of being bad credit risk for the data point with id 687?

Questions CO2 Dataset
- Can you describe the data?
- What is the average CO2 Emission of all countries?
- What are the most important features in predicting CO2 emission for Lebanon?
- How many countries have renewable energy consuption of less than 10?
- Can you list the top three important features for the model’s prediction in the data?
- What are the most important features for a country that produces no nuclear electricity?
- Assuming that the hydro electricity of Romania increases by 10, what will be the prediction for the CO2 emission?
- What is the average energy use for countries with GDP Per Capita greater then 10000?
- What are the top 5 features influencing the prediction for the country of armenia?
- What is the average CO2 emission of countries that have population density lower than 50 and GDP growth greater than 5?
- What country has the highest CO2 emissions and what are the main features affecting it?
- For countries with energy use greater than 2000 increase the renewable electricity by 10 and give explain how predictions changed?

These are the types of questions that can be asked. If you want the change the dataset you will have to restart the gradio interface, upload the new model and dataset
and fill in the respective boxes with the correct information given above. You may have to use the command 'ctrl + c' in the terminal to stop the interface so it can be rerun.

If you run into issues, here are some common solutions:
- Environment Errors: Make sure you've activated the correct environment (capstone_d2).
- Missing Dependencies: Ensure you've installed all the dependencies, including Gradio.
- API Key Issues: Double-check that your OpenAI API key is correct and active.
- Dataset or Model Issues: Ensure the correct .csv and .pkl files are uploaded.

Error in the chat interface: 
- Could be due to the LLM making a translation error because the exact match accuracy of its translations is not 100%, in this case you should be able to try a different question or reword the previous question in the textbox.
- The feature name or sample ID name used is not present in the dataset (eg. client with id 200 or country Armenia).
- The task you asked to be performed cannot be done with the current capabilities of the system.
