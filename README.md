This project predicts house prices in California. Upon cleaning the data, and performing feature engineering. A number of candidate models were trained and compared (XGBOOST, Linear Regression, Decision Tree, Random Forest) using cross-validation. XGBOOST performed best in terms of RMSE score and time taken to train. Hence it was chosen as the model to implement. First its hyperparameters were fine-tuned using RandomisedSearchCV. Which further improved the RMSE scores. A FastAPI was set up to allow for individual house price predictions. The project was containerised using Docker to allow for cross-compatibility.

While in the root folder.
To retrieve the dataset, split the data into train and test and clean the dataset - run "python main.py" in the terminal.
The data is explored by running python exploration.py .
The different ML models are compared by running python evaluate.py
The FastAPI is inialised by running python server.py
