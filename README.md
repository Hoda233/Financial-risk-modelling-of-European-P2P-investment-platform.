# Financial-risk-modelling-of-European-P2P-investment-platform

## About
About The main purposes of this analysis are to summarize the characteristics of variables that can affect the loan status and to get some ideas about the relationships among variables.

## Problem type
Supervised machine learning problem.

- Binary classification: 
  - Loan Status.
  
- Regression: 
  - EMI (Equated Monthly Installments)
  - ELA (Eligible Loan Amount)
  - ROI (Return on Investment)
  - ROI (Return on Investment)

## Libraries and packages
`conda install numpy` 

`conda install pandas`

`pip install seaborn`

`conda install matplotlib`

`conda install scikit-learn`

`conda install -c conda-forge xgboost`

` pip install streamlit`

## Structure 
- Loading Packages and Data
- Data Exploration
- Data Preprocessing
  - Drop unneeded features
  - For Classification: Handle target feature (Status)
  - For Regression: Generate targets features (EMI, ELA, ROI)
  - Handle missing values
  - Handle outliers
- EDA (Exploratory Data Analysis)
  - Univariate Analysis
  - Bivariate Analysis & Correlation Analysis
- Encoding Categorical Variables
- Scaling
- Feature Selection
- PCA (Principal Component Analysis)
- Separate Modeling 
  - For Classification:
    - Logistic Regression
    - Random Forest Classification
    - XGB Classification
    - Gaussian Naive Bayes 
  - For Regression:
    - Linear Regression
    - Ridge Regression
    - Random Forest Regression
## Final Pipelines
Each pipelines includes Scaling, PCA, Model.

- **Models used in Regression pipelines:**

| Model | R2 Scores for each target | R2 Score |
| ------| --------------------------| ---------|
| LinearRegression| [0.76497051 0.57446454 0.39988935]| 0.5797748017895326|
|Ridge| [0.7649705  0.57446456 0.39988934]| 0.5797748025381682|
|RandomForestRegressor| [0.92705843 0.71706524 0.8533169 ]| 0.8324801907433681|
|XGBRegressor| [0.93815903 0.71667844 0.85145203]| 0.8354298342198373|

- **Models used in Classification pipelines:**

| Model | R2 Scores for each target |
| ------| --------------------------|
| KNeighborsClassifier| 0.9311306028395153|
|XGBClassifier| 0.9449936817066825|
|LogisticRegression| 0.9276741247305433|
|RandomForestClassifier| 0.9417602021853861|

## Deployment

- Choose **XGBRegressor, XGBClassifier** as the final pipelines to be used in the app 
- Create a **streamlit app** to access model and get predictions

## How to run
- Clone the repo or download project folder
- Extract **Bondora_raw.zip** and put the data file in folder named **dataset**
- Run **Final_Pipeline.ipynb** to get the models
- To run the app: run `python -m streamlit run app.py` in vscode terminal 

