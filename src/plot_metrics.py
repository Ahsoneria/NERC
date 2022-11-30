import plotly.express as px
import pandas as pd


df = pd.DataFrame([['CRF', 99.8], ['BiLSTM', 99.06]], columns = ['Model Type' , 'Accuracy (%)'])


fig = px.bar(df, x='Model Type', y='Accuracy (%)',
             color='Accuracy (%)', width = 400 , height=400)
fig.update_layout(yaxis_range=[98,100])
fig.show()

df = pd.DataFrame([['CRF', 92.2], ['BiLSTM', 96.2] , ['BERT', 95.1]], columns = ['Model Type' , 'F1-Score (%)'])


fig = px.bar(df, x='Model Type', y='F1-Score (%)',
             color='F1-Score (%)', width = 500 , height=400)
fig.update_layout(yaxis_range=[80,100])
fig.show()

df = pd.DataFrame([['Recognize', 83.4], ['Label', 27.5] ], columns = ['Baseline Functionality' , 'F1-Score (%)'])


fig = px.bar(df, x='Baseline Functionality', y='F1-Score (%)',
             color='F1-Score (%)', width = 500 , height=400,  color_continuous_scale='Bluered_r')
fig.show()
