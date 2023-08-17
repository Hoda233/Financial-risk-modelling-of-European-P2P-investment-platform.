import streamlit as st
import numpy as np
import pickle as pkl


# load saved models
calssif_model = pkl.load(open("./models/xgb_classif.pkl", 'rb'))
reg_model     = pkl.load(open("./models/xgb_reg.pkl"    , 'rb'))

# define features

all_features = ['AppliedAmount' ,'BidsManual' ,'BidsPortfolioManager' ,'Country',
    'EmploymentStatus' ,'ExistingLiabilities' ,'ExpectedReturn' ,'IncomeTotal',
    'Interest' ,'InterestAndPenaltyBalance', 'InterestAndPenaltyPaymentsMade',
    'LanguageCode' ,'LiabilitiesTotal', 'Loan_Tenure' ,'LossGivenDefault',
    'MaritalStatus' ,'NewCreditCustomer', 'NoOfPreviousLoansBeforeLoan',
    'NrOfScheduledPayments', 'OccupationArea', 'PlannedInterestTillDate',
    'PreviousRepaymentsBeforeLoan' ,'PrincipalBalance', 'PrincipalPaymentsMade',
    'ProbabilityOfDefault' ,'Rating' ,'UseOfLoan']

features_regression=['AppliedAmount', 'BidsManual', 'BidsPortfolioManager', 'Country', 
                    'EmploymentStatus', 'ExistingLiabilities','ExpectedReturn', 'IncomeTotal', 
                    'Interest', 'InterestAndPenaltyBalance', 'InterestAndPenaltyPaymentsMade', 
                    'LanguageCode', 'LiabilitiesTotal', 'Loan_Tenure', 'LossGivenDefault', 
                    'NewCreditCustomer', 'NoOfPreviousLoansBeforeLoan',
                    'NrOfScheduledPayments', 'PlannedInterestTillDate', 'PreviousRepaymentsBeforeLoan', 
                    'PrincipalBalance', 'PrincipalPaymentsMade', 'ProbabilityOfDefault', 'Rating', 'UseOfLoan']

features_classification = ['InterestAndPenaltyBalance', 'PrincipalBalance', 'PrincipalPaymentsMade', 'ProbabilityOfDefault', 
                        'ExpectedReturn', 'LossGivenDefault', 'PlannedInterestTillDate', 'Interest', 'InterestAndPenaltyPaymentsMade', 
                        'AppliedAmount', 'Rating', 'NrOfScheduledPayments', 'LanguageCode', 'Country', 'BidsPortfolioManager', 
                        'IncomeTotal', 'OccupationArea', 'Loan_Tenure', 'PreviousRepaymentsBeforeLoan', 'MaritalStatus']

numerical_features = ['AppliedAmount' ,'BidsManual' ,'BidsPortfolioManager' ,
                        'ExistingLiabilities' ,'ExpectedReturn' ,'IncomeTotal',
                        'Interest' ,'InterestAndPenaltyBalance', 'InterestAndPenaltyPaymentsMade',
                        'LiabilitiesTotal', 'Loan_Tenure' ,'LossGivenDefault',
                        'NoOfPreviousLoansBeforeLoan','NrOfScheduledPayments',  'PlannedInterestTillDate',
                        'PreviousRepaymentsBeforeLoan' ,'PrincipalBalance', 'PrincipalPaymentsMade',
                        'ProbabilityOfDefault' ]

categorical_features = ['Country','EmploymentStatus' ,'LanguageCode' ,'MaritalStatus' ,'NewCreditCustomer',
                        'OccupationArea','Rating' ,'UseOfLoan']

# define mapping for categorical features

options = [
    {'EE':0, 'FI':1, 'ES':2, 'SK':3},
    {'Fully employed':0, 'Unknown':1, 'Partially employed':2, 'Self-employed':3,'Entrepreneur':4, 'Retiree':5},
    {'Estonian' :0,'Russian':1, 'English':2,'Finnish':3, 'Spanish':4, 'Other':5, 'Slovakian':6, 'German':7},
    {'Married':0, 'Divorced':1, 'Single':2, 'Cohabitant':3, 'Widow':4, 'Unknown':5},
    {'0':0,'1':1},
    {'Retail and wholesale': 0, 'Education': 1, 'Hospitality and catering': 2, 'Other': 3, 'Info and telecom': 4, 'Real-estate': 5, 
    'Transport and warehousing': 6, 'Construction': 7, 'Finance and insurance': 8, 'Healthcare and social help': 9, 'Utilities': 10,
    'Energy': 11, 'Processing': 12, 'Agriculture, forestry and fishing': 13, 'Unknown': 14, 'Art and entertainment': 15,
    'Civil service & military': 16, 'Research': 17, 'Administrative': 18, 'Mining': 19},
    {'E':0,'F':1,'C':2,'HR':3,'D':4,'B':5,'A':6,'AA':7},
    {'Other': 0, 'Home improvement': 1, 'Loan consolidation': 2, 'Vehicle': 3, 'Health': 4, 'Business': 5, 'Travel': 6, 
    'Education': 7, 'Real estate': 8, 'Other business': 9, 'Working capital financing': 10, 
    'Purchase of machinery equipment': 11, 'Accounts receivable financing': 12, 'Acquisition of real estate': 13, 
    'Construction finance': 14, 'Acquisition of stocks': 15, 'Unknown': 16},
]
    

def main():

    st.title("Financial-risk-modelling-of-European-P2P-investment-platform")
    
    features_values = {}
    values_reg = np.zeros(len(features_regression))
    values_classif = np.zeros(len(features_classification))

    for i in range(len(numerical_features)):
        feature_value = np.float64(st.number_input(f"{numerical_features[i]}: ", value=0.0, step=0.5, format="%.6f"))
        st.write('Value:', feature_value)
        features_values[numerical_features[i]] = feature_value
        
    
    for i in range(len(categorical_features)):
        selected_option = st.selectbox(f"{categorical_features[i]}: ", list(options[i].keys()))
        st.write('Choosen option:', selected_option)

        selected_option1 = options[i][selected_option]
        features_values[categorical_features[i]]  = selected_option1
        st.write('Encoded option', selected_option1)

    for i in range(len(features_regression)):
        values_reg[i]=features_values[features_regression[i]]

    for i in range(len(features_classification)):
        values_classif[i]=features_values[features_classification[i]]


    print(features_values,values_reg,values_classif)

    if st.button("Submit"):

        input_data_reg = values_reg.reshape(1, -1)
        input_data_clf = values_classif.reshape(1, -1)

        reg_pred = reg_model.predict(input_data_reg)
        classif_pred = calssif_model.predict(input_data_clf)

        st.write("EMI: ", reg_pred[0][0])
        st.write("ELA with EMI mean: ", reg_pred[0][1])
        st.write("ROI: ", reg_pred[0][2])
        st.write("Loan Status(default/non-default): ", classif_pred[0])

        
if __name__ == "__main__":
    main()