import yagmail
import pdfkit
import pandas as pd
import numpy as np
import tensorflow as tf
from numpy import argmax
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import load_model

import webbrowser
import os
import plotly.graph_objects as go
import plotly.io as pio


def Prediction(label_file):
    """"This function loads the model from Tensorflow and and does the prediction"""
    """Which is later displayed"""

    atks = {
        # 0: 'Benign',
        # 1: 'XSS',
        # 2: 'Local File Inclusion',
        # 3: 'SQL Injection',
        0: 'XSS',
        1: 'Benign',
        2: 'SQL Injection',
        3: 'Local File Inclusion',
    }

    df = pd.read_csv(label_file)

    ## need to change later to specific column
    data = pd.DataFrame(df)
    data = data["Logs"].values
    # print(data)

####
    #data3 = data["Logs"].values
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data)
    tfidf_matrix = vectorizer.transform(data)
    print("--------------1----------------")

    sparse_tensor = tf.sparse.SparseTensor(
        indices=np.mat([tfidf_matrix.nonzero()[0], tfidf_matrix.nonzero()[1]]).transpose(),
        values=tfidf_matrix.data.astype(np.float64),
        dense_shape=(589, 592))

    data2 = tf.sparse.reorder(sparse_tensor)
    print(data2.shape)
####

    # Run the current data through the model to get the predictions
    model = load_model("model/training_16.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    predictions = model.predict(data2)

    output_data = []
    output_data = []

    # Combines the dataset to display
    for p2, t, p3, p1 in zip(df['IP'].values, df['Timestamp'].values, df['Logs'].values, predictions):
        small_output = []
        small_output.append(p2)
        small_output.append(t)
        small_output.append(p3)
        small_output.append(atks[argmax(p1)])
        output_data.append(small_output)

    combination1 = pd.DataFrame(data=np.array(output_data), columns=['IP', 'Timestamp', 'Logs', 'Prediction'])


    return combination1

def create_pie_chart(df):
    values = df['Prediction'].value_counts()
    labels = values.index

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(title_text="Prediction Distribution")
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def display_and_export_html(df, file_name, filename='var/csv/html_output.html'):
    # Convert the DataFrame to an HTML table
    html_table = df.to_html(index=False, classes='table table-striped')
    print(html_table)
    pie_chart_html = create_pie_chart(df)

    # Get the current time and total predictions
    current_time = pd.Timestamp.now()
    now = current_time.strftime("%Y-%m-%d %H:%M")
    total_predictions = len(df) - len(df[df['Prediction'] == "Benign"])

    css_styles = '''
    .container {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    margin-top: 30px;
    margin-bottom: 30px;
    margin-left: auto;
    margin-right: auto;
    <!--padding: 30px;-->
                }
    h1 {
    color: #007bff;
        }
    table {
    table-layout: fixed;
    width: 100%
    }
    td {
     word-wrap: break-word;
    }
    '''

    html_content = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
            <style>
                {css_styles}
            </style>
        </head>
        <body>
            <div class="container">
                <h4>Report generated at {now} </h4>
                <h4>File Analysed: {file_name} </h4>
                <h1>Attack Statistics</h1>
                {pie_chart_html}
                <h4>Total Attacks Predicted: <b>{total_predictions}</b> </h4>
                <h1>Detailed Analysis</h1>
                {html_table}
            </div>
        </body>
    </html>
    """

    # Save the HTML content to a file
    with open(filename, 'w') as f:
        f.write(html_content)

    # config = pdfkit.configuration(wkhtmltopdf=r"C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe")
    # pdfname = 'var/csv/html_output.pdf'
    # pdfkit.from_file('output.html', 'html_output.pdf', configuration=config)

    # Display the HTML content in a web browser
    # webbrowser.open('file://' + os.path.realpath(filename))

    # yagmail.register('johnnydowwe2206@gmail.com', 'rhyyowncoqxqwdad')
    # johnnydowwe2206@gmail.com, Ict2206!, 01Jan2000

    # receiver = "icekenneth@hotmail.com"
    receiver = "shathiyasulthana@outlook.com"
    body = "Your threat report is here."
    filename = filename

    yag = yagmail.SMTP("johnnydowwe2206@gmail.com")
    yag.send(
        to=receiver,
        subject="Yagmail test with attachment",
        contents=body,
        attachments=filename,
    )

