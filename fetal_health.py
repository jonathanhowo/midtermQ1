# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Fetal Health Classification: A Machine Learning App') 
st.write('Utilize our advanced Machine Learning application to predict fetal health classfications.')
# Display an image of penguins
st.image('fetal_health_image.gif', width = 400)

# Load the pre-trained model from the pickle file
dt_pickle = open('decision_tree.pickle', 'rb') 
clf = pickle.load(dt_pickle) 
dt_pickle.close()

rf_pickle = open('random_forest.pickle', 'rb')
clf2 = pickle.load(rf_pickle)
rf_pickle.close()

ab_pickle = open('ada_boost.pickle', 'rb') 
clf3 = pickle.load(ab_pickle) 
ab_pickle.close()

sv_pickle = open('soft_voting.pickle', 'rb')
clf4 = pickle.load(sv_pickle)
sv_pickle.close()


# Create a sidebar for input collection
st.sidebar.subheader('Fetal Health Features Input')

dff = pd.read_csv('fetal_health.csv').drop(columns=['fetal_health']).dropna()

file_upload = st.sidebar.file_uploader("Upload your data", type=['csv'])
st.sidebar.warning(body = " *Ensure your data exactly matches the format outlined below*", icon = "⚠️")
st.sidebar.write(dff.head())
option = st.sidebar.radio("Choose Model for Prediction", ("Random Forest", "Decision Tree", "AdaBoost", "Soft Voting"))


st.sidebar.info(body = f" *You selected: {option}*", icon="✅")

if file_upload is None:
    st.info('Please upload to proceed')
else:
    st.success('CSV file uploaded successfully')

    user_df = pd.read_csv(file_upload).dropna() 
    if option == "Decision Tree":
        clf_type = clf
        feature_imp = 'feature_imp.svg'
        class_report = 'class_report.csv'
        confusion_mat = 'confusion_mat.svg'
    elif option == "Random Forest":
        clf_type = clf2  
        feature_imp = 'feature_imp2.svg'
        class_report = 'class_report2.csv'
        confusion_mat = 'confusion_mat2.svg'
    elif option == "AdaBoost":
        clf_type = clf3
        feature_imp = 'feature_imp3.svg'
        class_report = 'class_report3.csv'
        confusion_mat = 'confusion_mat3.svg'
    else:
        clf_type = clf4
        feature_imp = 'feature_imp4.svg'
        class_report = 'class_report4.csv'
        confusion_mat = 'confusion_mat4.svg'     




    #used ai to help figure this function out      
    def colored_background(val):
        color = ''
        if val == 'Normal':
                color = 'background-color: lime; color: white;' 
        elif val == 'Suspect':
                color = 'background-color: yellow; color: black;'  
        elif val == 'Pathological':
                color = 'background-color: orange; color: white;' 
        return color

    predictions = clf_type.predict(user_df)


    prob_scores = clf_type.predict_proba(user_df)


    user_df['Predicted Fetal Health'] = predictions
    user_df['Prediction Probability (%)'] = np.max(prob_scores, axis = 1) * 100
    user_df['Prediction Probability (%)'] = user_df['Prediction Probability (%)'].apply(lambda x: f"{x:.1f}")
    
    final_df = user_df.style.applymap(colored_background, subset=['Predicted Fetal Health'])
    st.header(f'Predicting Fetal Health Class Using {option} Model')
    st.write(final_df)
    st.header("Model Performance and Insights")

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", 
                                "Classification Report", 
                                "Feature Importance"])

    with tab1:
            st.write("### Confusion Matrix")
            st.image(confusion_mat)
            st.caption("Confusion Matrix of model predictions.")
    with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv(class_report, index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1 score, and Support for each health condition.")
    with tab3:
            st.write("### Feature Importance")
            st.image(feature_imp)
            st.caption("Features used in this prediction are ranked by relative importance.")

