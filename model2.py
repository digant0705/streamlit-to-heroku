# app1.py
import streamlit as st
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC


def app():
    
    siteHeader = st.beta_container()
    dataExploration = st.beta_container()
    modelTraining = st.beta_container()
    
    @st.cache
    def getData(train,test):
        df_train = pd.read_csv(train)
        df_test=pd.read_csv(test)
        return (df_train,df_test)
    
    
    with siteHeader:
        st.title('Logistic Regression')
        # st.text('In this project I look into ... And I try ... I worked with the dataset from ...')
        html_temp = """
            <div style ="background-color:powderblue;padding:13px">
            <h1 style ="color:black;text-align:center;">Data Description </h1>
            <p>In this project We work on the dataset that contains leads collected by an
            online education portal called \'X Education\'. We will train on '80%' of the data and test on the remaining
            '20%' data. Some of the rows of the training and testing dataset are shown below:
            </p>
            </div>
                    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
        st.markdown(html_temp, unsafe_allow_html = True)
    
    with dataExploration:
        st.header('Dataset: Leads conversion dataset')
        df_train,df_test = getData('data/train_lead_conversion_after_rfe.csv','data/test_lead_conversion_after_rfe.csv')

        
        html_temp = """
            <div style ="background-color:white;padding:10px">
            <center><label>Train Data</label></center>
            </div>
                    """
        st.markdown(html_temp, unsafe_allow_html = True)

        st.write(df_train.head())

        
        html_temp = """
            <div style ="background-color:white;padding:10px">
            <center><label>Test Data</label></center>
            </div>
                    """
        st.markdown(html_temp, unsafe_allow_html = True)

        
        st.write(df_test.head())
    
    
    
    with modelTraining:
        st.header('Model training')
        st.text('In this section you can select the hyperparameters! and view the performance of model')
    
        selection_col, display_col = st.beta_columns(2)

        max_iter= selection_col.slider('Convergence parameter !', 
          min_value=100, 
          max_value=1000, 
          value=600, 
          step=100)
    
        C = selection_col.slider('What should be the C (inverse of regularization) of the model?', 
          min_value=1.0, 
          max_value=10.0, 
          value=1.0, 
          step=1.0)
        
        solver = selection_col.radio("Choose Solver", ['newton-cg','lbfgs','liblinear'])

        selection_col.text('Here is a list of features: ')
        selection_col.write(df_train.columns)
    
        #for train data
        X_train_rf = df_train.drop([ "Converted"], 1)
        y_train = df_train["Converted"]
    
        #for test data
        X_test_rf = df_test.drop([ "Converted"], 1)
        y_test = df_test["Converted"]
    
        # code for modeling
        sgd= LogisticRegression(max_iter=max_iter,C=C,solver=solver)
        sgd.fit(X_train_rf,y_train)
    
        #train data predict
        prediction = sgd.predict(X_train_rf)
        display_col.subheader('Mean absolute error for train data:') 
        display_col.write(mean_absolute_error(y_train, prediction))
    
        display_col.subheader('Confusion Metrics for train data:') 
        confusion = metrics.confusion_matrix(y_train,prediction.round())
        display_col.write(confusion)
    
        display_col.subheader("Model accuracy for train data:")
        display_col.write(metrics.accuracy_score(y_train,prediction.round()))
    
        #test data predict
        prediction = sgd.predict(X_test_rf)
        display_col.subheader('Mean absolute error for test data:') 
        display_col.write(mean_absolute_error(y_test, prediction))
    
        display_col.subheader('Confusion Metrics for test data:') 
        confusion = metrics.confusion_matrix(y_test,prediction.round())
        display_col.write(confusion)
    
        display_col.subheader("Model accuracy for test data:")
        display_col.write(metrics.accuracy_score(y_test,prediction.round()))
        
    
        
        
        y_test_pred_prob=sgd.predict_proba(X_test_rf)
        y_test_pred_prob_pos= y_test_pred_prob[:,1]
        
        y_test_pred_final = pd.DataFrame({'Converted':y_test, 'Predicted': prediction, 'Conversion_Score':y_test_pred_prob_pos*100})
        y_test_pred_final['LeadID'] = y_test.index
        y_test_pred_final=pd.DataFrame.sort_values(y_test_pred_final, by= 'Conversion_Score',ascending=False)
        
        html_temp = """
            <div style ="background-color:white;padding:10px">
            <center><label>Conversion Scoring</label></center>
            </div>
                    """
        st.markdown(html_temp, unsafe_allow_html = True)

        st.write(y_test_pred_final.head(50))
    