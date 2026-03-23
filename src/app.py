import streamlit as st
import pickle
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import plotly.graph_objects as go

# Load the trained model
model_rf = pickle.load(open('models/random_forest_model.pkl', 'rb'))
model_svm = pickle.load(open('models/svm_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

st.title('Gamma Ray Classification')

st.write("""
The MAGIC (Major Atmospheric Gamma Imaging Cherenkov) telescope
detects high-energy gamma rays from space. This app classifies
telescope events as either **Gamma** (signal) or **Hadron** (background noise)
based on 10 shape features of the light pattern captured by the telescope.

The dataset contains 19,020 events collected from the MAGIC telescope
and is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope).
""")

#Model selection
model_option = st.selectbox('Select the model you want to use for prediction:', ('Random Forest', 'SVM'))

# Input fields for the features
fLength  = st.slider('fLength (Major axis length)',  min_value=4.28,   max_value=334.18, value=50.0)
fWidth   = st.slider('fWidth (Minor axis length)',   min_value=0.0,    max_value=256.38, value=20.0)
fSize    = st.slider('fSize (Log of pixel content)', min_value=1.94,   max_value=5.32,   value=2.5)
fConc    = st.slider('fConc (Brightest pixels ratio)',min_value=0.013, max_value=0.893,  value=0.3)
fConc1   = st.slider('fConc1 (Brightest pixel ratio)',min_value=0.0,   max_value=0.675,  value=0.2)
fAsym    = st.slider('fAsym (Asymmetry)',             min_value=-457.0, max_value=770.0,  value=0.0)
fM3Long  = st.slider('fM3Long (3rd moment long)',    min_value=-331.0, max_value=527.0,  value=0.0)
fM3Trans = st.slider('fM3Trans (3rd moment trans)',  min_value=-205.0, max_value=179.0,  value=0.0)
fAlpha   = st.slider('fAlpha (Angle of major axis)', min_value=0.0,    max_value=90.0,   value=20.0)
fDist    = st.slider('fDist (Distance from origin)', min_value=1.26,   max_value=457.0,  value=100.0)

# Predict button
if st.button('Predict'):
    # Create a feature array from the input values
    features = np.array([[fLength, fWidth, fSize, fConc, fConc1,
                          fAsym, fM3Long, fM3Trans, fAlpha, fDist]])
    
    # Make prediction based on the selected model
    if model_option == 'Random Forest':
        prediction = model_rf.predict(features)
        proba = model_rf.predict_proba(features)
        g_index = list(model_rf.classes_).index('g')
    else:
        features_scaled = scaler.transform(features)
        prediction = model_svm.predict(features_scaled)
        proba = model_svm.predict_proba(features_scaled)
        g_index = list(model_svm.classes_).index('g')

    confidence = proba[0][g_index] if prediction[0] == 'g' else 1 - proba[0][g_index]

    # Display the prediction result
    if prediction[0] == 'g':
        st.success(f'The event is classified as Gamma Ray ({confidence:.1%} confident).')
    else:
        st.error(f'The event is classified as Hadron ({confidence:.1%} confident).')

# Model performance tables
st.subheader('Model Performance')

magic = fetch_ucirepo(id=159)
X = magic.data.features
y = magic.data.targets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_scaled = scaler.transform(X_test)
y_true = y_test.values.ravel()

y_pred_rf = model_rf.predict(X_test)
y_pred_svm = model_svm.predict(X_test_scaled)

def get_metrics(y_true, y_pred, name):
    return {
        'Model': name,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, pos_label='g'), 4),
        'Recall': round(recall_score(y_true, y_pred, pos_label='g'), 4),
        'F1 Score': round(f1_score(y_true, y_pred, pos_label='g'), 4),
    }

metrics_df = pd.DataFrame([
    get_metrics(y_true, y_pred_rf, 'Random Forest'),
    get_metrics(y_true, y_pred_svm, 'SVM'),
])
st.dataframe(metrics_df, hide_index=True)

# Dataset table
st.subheader('Dataset')
st.dataframe(X.join(y))

# ROC Curve
st.subheader('ROC Curve')
y_test_binary = (y_true == 'g').astype(int)

g_index_rf = list(model_rf.classes_).index('g')
y_prob_rf = model_rf.predict_proba(X_test)[:, g_index_rf]
fpr_rf, tpr_rf, _ = roc_curve(y_test_binary, y_prob_rf)
auc_rf = auc(fpr_rf, tpr_rf)

g_index_svm = list(model_svm.classes_).index('g')
y_prob_svm = model_svm.predict_proba(X_test_scaled)[:, g_index_svm]
fpr_svm, tpr_svm, _ = roc_curve(y_test_binary, y_prob_svm)
auc_svm = auc(fpr_svm, tpr_svm)

fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', name=f'Random Forest (AUC = {auc_rf:.2f})'))
fig.add_trace(go.Scatter(x=fpr_svm, y=tpr_svm, mode='lines', name=f'SVM (AUC = {auc_svm:.2f})'))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
st.plotly_chart(fig)

# Confusion matrices
st.subheader('Confusion Matrix')
col1, col2 = st.columns(2)

for col, y_pred, name in [(col1, y_pred_rf, 'Random Forest'), (col2, y_pred_svm, 'SVM')]:
    cm = confusion_matrix(y_true, y_pred, labels=['g', 'h'])
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=['Predicted Gamma', 'Predicted Hadron'],
        y=['Actual Gamma', 'Actual Hadron'],
        colorscale='Blues', text=cm, texttemplate='%{text}'
    ))
    fig_cm.update_layout(title=name)
    col.plotly_chart(fig_cm, use_container_width=True)

# Class distribution
st.subheader('Class Distribution')
class_counts = y['class'].value_counts().rename(index={'g': 'Gamma', 'h': 'Hadron'})
fig_dist = go.Figure(go.Bar(x=class_counts.index, y=class_counts.values))
fig_dist.update_layout(xaxis_title='Class', yaxis_title='Count')
st.plotly_chart(fig_dist)
