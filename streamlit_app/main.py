# -*- coding: utf-8 -*-

# streamlit run C:\Users\theop\Desktop\streamlit_app\main.py

import sys
import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from random import randint

#%% Functions
@st.cache()

# Load the subset which is in the folder where y is the target variable
def load_test_subset(path):
    file_x = path + "test_subset/x_test_df.csv"
    file_y = path + "test_subset/y_test_df.csv"
    x_test = pd.read_csv(file_x,header=0)
    y_test = pd.read_csv(file_y,header=0)
    return x_test, y_test


# Make predictions on the subset with the model, after dimensionality reduction (PCA) 
def prediction_subset(classifier, pca, path):
    x_test, y_test = load_test_subset(path)
    
    x_test_pca = pca.transform(x_test)
    
    preds = classifier.predict(x_test_pca)
        
    return preds, y_test, x_test

# Convert the music name to a DataFrame (90 predictors : Avg1 to 12, Cov1 to 78)
def music_name_to_data(music, path):
    file = path + "musics/"
    
    if music == "Seal - Crazy":
        file += "seal_crazy.ent12"
    elif music == "Oasis - Wonderwall":
        file += "oasis_wonderwall.ent12"
    elif music == "Maroon 5 - This Love":
        file += "maroon5_thislove.ent12"
    
    # get the data from the folder and put it in dataframe 
    # each row is a timbre representation, in 12 dimensions (12 columns)
    col = [f"C{i}" for i in range(1,13)]
    data = pd.read_csv(file, sep=',', names=col).drop(index=0)
    data.C1 = pd.to_numeric(data.C1)
    
    
    # computing the 12 Avg variables (average of each of the 12 columns)
    d_avg = dict()
    
    for i in range(1,13):
      d_avg[f"Avg{i}"] = data[f"C{i}"].mean()
    
    df_avg = pd.DataFrame(pd.Series(d_avg)).transpose()
    
    triu = pd.DataFrame(np.triu(data.cov().to_numpy()))
    j = 0
    d = []
    for i in range(12):
      d.extend([triu[k][i] for k in range(0+j,12)])
      j+=1
    
    # computing the 78 Cov variables (upper triangular part of the covariance matrix)
    colCOV = [f"Cov{i}" for i in range(1,79)]
    
    df_cov = pd.DataFrame(d).transpose()
    df_cov.columns = colCOV
    
    new = pd.concat([df_avg,df_cov],axis=1)
    
    return data, new


# Make prediction of the music name given : PCA then predict with model
def prediction(music, classifier, pca, path):    
    _, df = music_name_to_data(music, path)
    
    df_pca = pca.transform(df)
    
    pred = classifier.predict(df_pca)
    pred_final = [round(p) for p in pred]
        
    return df_pca, pred_final[0]


# Option of the selection box : Test the model on a new song
def test_new_song(classifier, pca, path):
    st.markdown(""" 
    <div style ="background-color:#e6e6fa;padding:13px"> 
    <h5 style ="color:black;text-align:center;">Pick a song to predict its release year</h5> 
    </div> 
    """,  unsafe_allow_html=True)
    
    # Music selection
    music = st.radio("", ("Maroon 5 - This Love", "Oasis - Wonderwall", "Seal - Crazy"))
    res = 0
    year = 0
    
    if music == "Oasis - Wonderwall":
        st.video("https://www.youtube.com/watch?v=6hzrDeceEKc")
        res = 1995 # real release date of the music
        
    elif music == "Seal - Crazy":
        st.video("https://www.youtube.com/watch?v=4Fc67yQsPqQ")
        res = 1991
        
    elif music == "Maroon 5 - This Love":
        st.video("https://www.youtube.com/watch?v=XPpTgCho5ZA")
        res = 2002
    
    # After selection of the music, make prediction and then show the results
    if(st.button("Predict it !")):
        st.write(f"Expected : {res}")

        _, year = prediction(music, classifier, pca, path)
        diff = abs(res-year)
        st.success(f"Predicted : {year}")

        st.write(f"Difference of {diff} years")
    
    st.title("")
    
    # Show representations of the selected music : its timbre representation and its Avg&Cov representation
    if(st.button("Show its mathematical representation")):
        data, df = music_name_to_data(music, path)
        
        st.title("")
        st.markdown(""" 
        <div style ="background-color:#e6e6fa;padding:13px"> 
        <h5 style ="color:black;text-align:center;">Representation of its timbres, given by the Echo Nest API</h5> 
        </div> 
        """,  unsafe_allow_html=True)        
        st.write("Each row is a 12-dimensional vector that represents a segment of the music")
        st.write("Each dimension is part of a linear combination which discribes the timbre of this segment")
        st.write("For example, dim1 is related to loudness, dim2 to brightness, dim3 to flatness, dim4 to attack, etc.")
        st.write(data)
        
        st.title("")
        st.markdown(""" 
        <div style ="background-color:#e6e6fa;padding:13px"> 
        <h5 style ="color:black;text-align:center;">Averages and Covariances of these timbres</h5> 
        </div> 
        """,  unsafe_allow_html=True) 
        st.write("After mathematical computations, I obtain this data which is used by the model to predict its release year.")
        st.write(df.iloc[0])
    
    
# Option of the selection box : Test the model of a new-created song 
def test_chosen_data(classifier, pca):
    st.title("")
    st.markdown(""" 
    <div style ="background-color:#e6e6fa;padding:13px"> 
    <h5 style ="color:black;text-align:center;">Here, you can enter the values of your choice to represent a "song" </h5> 
    <h5 style ="color:black;text-align:center;">New random values are automatically set every time you click the button "Predict !"</h5> 
    <h5 style ="color:black;text-align:center;">The button is at the bottom of the window</h5> 
    <h5 style ="color:black;text-align:center;">Don't hesitate to try it many times !</h5> 
    <h5 style ="color:black;text-align:center;">NB : You don't create a really song by doing this, it's just an example.</h5> 
    <h5 style ="color:black;text-align:center;">But, the range for each predictor is the real one. The only difference is that you can only select integers here.</h5> 

    </div> 
    """,  unsafe_allow_html=True)
    st.title("")
    
    # Display the range of each predictors with sliders in order to make the user choose the settings that he wants
    slider_dic = {'Avg1': [2, 62], 'Avg2': [-337, 384], 'Avg3': [-301, 323], 'Avg4': [-154, 336], 'Avg5': [-182, 262], 'Avg6': [-82, 166], 'Avg7': [-188, 172], 'Avg8': [-73, 127], 'Avg9': [-126, 146], 'Avg10': [-42, 60], 'Avg11': [-70, 88], 'Avg12': [-94, 88], 'Cov1': [0, 550], 'Cov2': [8, 65736], 'Cov3': [21, 36817], 'Cov4': [18, 31849], 'Cov5': [12, 19866], 'Cov6': [6, 16832], 'Cov7': [20, 11902], 'Cov8': [6, 9570], 'Cov9': [6, 9617], 'Cov10': [15, 3722], 'Cov11': [6, 6737], 'Cov12': [5, 9813], 'Cov13': [-2821, 2050], 'Cov14': [-13390, 24480], 'Cov15': [-12017, 14505], 'Cov16': [-4325, 3411], 'Cov17': [-3357, 3278], 'Cov18': [-3115, 3553], 'Cov19': [-3806, 2347], 'Cov20': [-1516, 1954], 'Cov21': [-1679, 2888], 'Cov22': [-1591, 2330], 'Cov23': [-990, 1813], 'Cov24': [-1711, 2496], 'Cov25': [-8448, 14149], 'Cov26': [-10096, 8059], 'Cov27': [-9804, 6065], 'Cov28': [-7883, 8360], 'Cov29': [-4673, 3538], 'Cov30': [-4175, 3892], 'Cov31': [-4975, 1202], 'Cov32': [-1073, 1831], 'Cov33': [-1021, 747], 'Cov34': [-1330, 1199], 'Cov35': [-14862, 9060], 'Cov36': [-3993, 6968], 'Cov37': [-6642, 6172], 'Cov38': [-2345, 2067], 'Cov39': [-2271, 1427], 'Cov40': [-1746, 2460], 'Cov41': [-3188, 2395], 'Cov42': [-2200, 2901], 'Cov43': [-1694, 569], 'Cov44': [-5154, 6955], 'Cov45': [-5112, 12700], 'Cov46': [-4731, 13001], 'Cov47': [-3756, 5419], 'Cov48': [-2500, 5690], 'Cov49': [-1900, 1811], 'Cov50': [-1397, 973], 'Cov51': [-600, 812], 'Cov52': [-10346, 11048], 'Cov53': [-7376, 2878], 'Cov54': [-3896, 3447], 'Cov55': [-1199, 2055], 'Cov56': [-2565, 4780], 'Cov57': [-1905, 5287], 'Cov58': [-975, 746], 'Cov59': [-7058, 3958], 'Cov60': [-6953, 4741], 'Cov61': [-8401, 2124], 'Cov62': [-1813, 1640], 'Cov63': [-1388, 1278], 'Cov64': [-718, 741], 'Cov65': [-9831, 10020], 'Cov66': [-2026, 3424], 'Cov67': [-8390, 5188], 'Cov68': [-4755, 3735], 'Cov69': [-438, 841], 'Cov70': [-4402, 4469], 'Cov71': [-1811, 3211], 'Cov72': [-3098, 1734], 'Cov73': [-342, 261], 'Cov74': [-3169, 3662], 'Cov75': [-4320, 2834], 'Cov76': [-236, 463], 'Cov77': [-7458, 7393], 'Cov78': [-381, 678]}
    inputs = []
    for k in slider_dic.keys():
        mini = slider_dic[k][0]
        maxi = slider_dic[k][1]
        # Random settings are already put
        inputs.append(st.slider(k,mini,maxi,randint(mini,maxi)))
    
    # Button to predict the new-created song and show the result
    st.title("")
    st.title("")
    if (st.button("Predict !")):        
        inputs_pca = pca.transform([inputs])      
        pred = classifier.predict(inputs_pca)
        pred_final = [round(p) for p in pred]
        st.success(pred_final[0])
    
    
# Option of the selection box : Visualization of the dataset
def data_viz(path):
    path_images = path + "images/"
    
    # Show a subset of the dataset 
    st.header("The dataset")
    st.write("Each row represent a song - There are 90 predictors - Year is the target variable")
    x, y = load_test_subset(path)
    st.write(pd.concat([y,x],axis=1))
    
    # Show many different viz
    st.header("Dataset visualizations") 
    songs_year = Image.open(path_images + "songs_year.png")
    st.image(songs_year,output_format="PNG", use_column_width=False)
    songs_under_year = Image.open(path_images + "songs_under_year.png")
    st.image(songs_under_year,output_format="PNG", use_column_width=False)

    st.header("PCA visualizations")
    st.subheader("2D PCA")
    pca2D = Image.open(path_images + "pca2Dsolo.png")
    st.image(pca2D,output_format="PNG", use_column_width=False, caption="Every songs for each decade is plotted on 2 dimensions")
    st.subheader("3D PCA")
    pca3D = Image.open(path_images + "pca3Dcombo.png")
    st.image(pca3D,output_format="PNG", use_column_width=False, caption="Every songs for each decade is plotted on 3 dimensions")

    st.header("Mean of the values of predictors depending on years")    
    st.subheader("Averages predictors")
    avg_pred = Image.open(path_images + "avg_var.png")
    st.image(avg_pred,output_format="PNG", use_column_width=False)
    st.subheader("Covariances predictors")
    cov_pred = Image.open(path_images + "cov_var.png")
    st.image(cov_pred,output_format="PNG", use_column_width=False)
    
    
# Option of the selection box : Check the model performance
def model_perf(classifier, pca, path):
    path_images = path + "images/"
    
    st.subheader("Model : 3-NN")
    st.write("KNighborsClassifier(n_neighbors=3, weights='distance')")
    st.write("K-NN with k = 3 where each neighbors is weighted by its distance")
    st.write("On the plots, 3-NN model is on the left")
    st.write("All the explanations are available on my notebook")
    st.title("")
    
    # Make predictions with the model on the subset and show them in a DataFrame if button 
    st.subheader("Let the model predict the following songs : ")
    preds, y_test, x_test = prediction_subset(classifier, pca, path)
    df_test = pd.concat([y_test[:9],x_test[:9]],axis=1)
    st.write(df_test)
    
    if (st.button("Predict !")):
        df_fin = pd.concat([y_test[:9],pd.Series(preds[:9])],axis=1)
        df_fin.columns = ["Real","Predicted"]
        st.write(df_fin)
    
    # Show different viz of metrics of our model compared to other one
    acc = Image.open(path_images + "acc_models.png")
    st.image(acc,output_format="PNG", use_column_width=False)
    diff_year = Image.open(path_images + "avg_abs_diff_years_models.png")
    st.image(diff_year,output_format="PNG", use_column_width=False)
    diff_nb = Image.open(path_images + "avg_diff_nb_models.png")
    st.image(diff_nb,output_format="PNG", use_column_width=False)


#%% Main

def main(model = "model_final.pkl"):
    
    # Take the path where the folder of the whole API is
    path = str(sys.argv[0])[:-7].replace(os.sep,'/')
    
    # Load the model
    path_model = path + "models/" + model
    pickle_in = open(path_model, "rb")
    classifier = pickle.load(pickle_in)
    
    # Load the PCA (55 components)
    pca = pickle.load(open(path + "models/pca.pkl", "rb"))
    
    # Front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:Lavender;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Year Prediction with a subset of the Million Song Dataset</h1> 
    </div> 
    
    <div style ="background-color:#e3d1f6;padding:13px"> 
    <h2 style ="color:black;text-align:center;">Predict the release year of a song based on audio features</h2> 

    </div> 
    """
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #f0e6fa;position:relative;color:black;font-size:20px;width:20em;border-radius:10px 10px 10px 10px;
    }
    </style>""", unsafe_allow_html=True)  
        
    # Display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    st.caption("")
    st.caption("By Théophane DELBAUFFE")
    st.caption("Master's Degree in Data & Artificial Intelligence")
    st.caption("ESILV - Engineering School - France")

    st.title("")
    st.title("")
    
    st.markdown(""" 
    <div style ="background-color:#f0e6fa;padding:13px"> 
    <h3 style ="color:black;text-align:center;">What do you want to do ?</h3> 
    </div> 
    """,  unsafe_allow_html=True)
    window = st.selectbox("", 
                          ["- Pick an option -", "See dataset visualizations", "Check model performance", "Test the model with a new song", "Test the model with values of your choice"])
    
    
    # The user must select an option to see more 
    if window == "See dataset visualizations":
        data_viz(path)
    
    elif window == "Check model performance":
        model_perf(classifier, pca, path)
    
    elif window == "Test the model with a new song":      
        test_new_song(classifier, pca, path)
    
    elif window == "Test the model with values of your choice":
        test_chosen_data(classifier, pca)


#%% 

if __name__ == '__main__':
    main()

