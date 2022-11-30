NERC : Named Entity Recognition and Classification

We have created an User Interface (UI), which takes input English text, and outputs multiple classifications for the text.

We recognize and classify ```Person, Location, Organization, Entity```.

We utilize 5 Machine Learning Models : ```CRF, BERT, SpaCy, BiLSTM+CRF, Baseline``` models.

To run the UI, in Google Colab, run following snippet:
(tokens edited)

```
!pip install waitress
!pip install flask-ngrok
!pip install pyngrok==4.1.1
!pip install transformers
!pip install keras
!pip install sklearn-crfsuite
!pip install datasets
!python -m pip install markupsafe==2.0.1
!pip install git+https://www.github.com/keras-team/keras-contrib.git

!ngrok authtoken [self_ngrok_token]

!git clone https://[github_personal_token]@github.com/Ahsoneria/NERC.git

!python3 NERC/src/main.py
```

In ```src``` directory following files are present:

  &nbsp;&nbsp;&nbsp;&nbsp;```main.py```: Flask routing file. This is the file which is the entry-point to the UI.
  
  &nbsp;&nbsp;&nbsp;&nbsp;```api.py```: Consists of all API calls : 
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; data preprocessing steps, 
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; getting entities and their labels for each type of ML model (SpaCy, BERT, CRF, Baseline)
  
 &nbsp;&nbsp;&nbsp;&nbsp; ```templates/index.html```: Home page, along with important CSS styling and JavaScript highlighting functions.
 
 &nbsp;&nbsp;&nbsp;&nbsp; ```templates/result.html```: Page with divs for Display of multiple results along with the Legend of color scheme for highlighting.

 &nbsp;&nbsp;&nbsp;&nbsp; ```Final_CRF.ipynb```: Jupyter notebook to be run on google colab to train the CRF model.
  
 &nbsp;&nbsp;&nbsp;&nbsp; ```crf.pkl```: Saved CRF model which is directly used by corresponding API, so no re-training required.
  
 &nbsp;&nbsp;&nbsp;&nbsp; ```plot_metrics.py```: Generates plots used in presentation.
  
  Because of TensorFlow version conflicts (2.x vs 1.y), the BiLSTM model could not be currently integrated into the UI with the others.
  
  &nbsp;&nbsp;&nbsp;&nbsp;```bilstm.py```: Consists of the bilstm class, to perform NERC using the BiLSTM + CRF model.
  
 &nbsp;&nbsp;&nbsp;&nbsp; ```bilstm_requirements.txt```: To install requirements only to run BiLSTM + CRF model ```pip install src/bilstm_requirements.txt```
  
  &nbsp;&nbsp;&nbsp;&nbsp;```ner_dataset.csv```: For training BiLSTM + CRF model
  
  


