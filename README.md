# ML-YearPredictionMSD

## The project
The goal of this Machine Learning project is to build a model to **predict the release year of a song**, based on **timbre audio features**.

Data was collected by the **Echo Nest API**, which loads music files and returns a JSON file containing much information. 

The dataset consists of **515345 instances**, 90 predictors and 1 target : Year. 

My final model is : `KNeighborsClassifier(n_neighbors=3, weights='distance')`

It got an accuracy of 5.7 % and an accuracy to the decade of 47.7 %.

Its average absolute distance from the original release year is 9 years.

## The repository contains
- A **Jupyter Notebook** where you can find all data processing, dataviz and machine learning algorithms
- An **API**
- A **PDF** file that summarizes my work
- Documentation about the subject, the context and the dataset 

## Launching the API
- Download the *streamlit_app* file
- Download the **fitted model** here : https://drive.google.com/file/d/10WuNKLmeB1-VEW5MpHm7vBf2FiyVpevd/view?usp=sharing
- Or download the dataset here : https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
  - Then download the Jupyter Notebook
  - Train the model by yourself 
  - Save it in a pkl format
- Put the **model_final.pkl** in *streamlit_app/models/*
- Install **streamlit library** : `pip install streamlit`
- Run the API on **Anaconda Prompt** or else with this command :
  - `streamlit run ...\streamlit_app\main.py`
  - For example, I enter this command : `streamlit run C:\Users\theop\Desktop\streamlit_app\main.py`
- Go to http://localhost:8501
