#from turtle import width
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, classification_report, plot_roc_curve
from sklearn.metrics import accuracy_score
import streamlit as st

st.title('Fraud Detection System')
df = pd.read_csv("creditcard.csv")
model = pickle.load(open('model.pkl', 'rb'))
X_test=pickle.load(open('xtest.pkl','rb'))
y_test=pickle.load(open('ytest.pkl','rb'))
X_test_prediction=pickle.load(open('xtestpred.pkl','rb'))


def preda(df2):
    df2 = df2.drop(columns=['Class', 'account'], axis=1)
    df2pred = model.predict(df2)
    if (df2pred == 0):
        return 0
    else:
        return 1


df['account'] = np.arange(len(df))
List = list(df['account'])
cols = list(df.columns)
cols = [cols[-1]] + cols[:-1]
df = df[cols]

st.subheader("The transactions of accounts :")
st.dataframe(df)

number = st.text_input('Select  the account you want check for Fraud',0)
df2 = df.loc[(df['account'] == int(number))]
if st.button('Check for Fraud'):

    st.text("Account Transactions Details :")
    st.table(df2)

    import time

    my_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.001)
        my_bar.progress(percent_complete + 1)

    if ((preda(df2)) == 0):
        st.success("Not Fraud")
        #st.balloons()
    else:
        st.error("Fraud!!!")

st.title("  ")
st.title("  ")
st.header("Model Characteristic :")
st.subheader("Classification Report:")
st.text(classification_report(X_test_prediction, y_test))

st.subheader ("Accuracy Score on test data :")
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)
st.write(test_data_accuracy)

st.set_option('deprecation.showPyplotGlobalUse', False)
plot_confusion_matrix(model, X_test, y_test)
plt.title('Confusion Matrix\n')
fig=plt.show()
st.pyplot(fig)

plot_roc_curve(model, X_test, y_test)
plt.title('ROC-AUC-CURVE\n')
fig2=plt.show()
st.pyplot(fig2)