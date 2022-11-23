
In Google Colab, run following snippet:
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
