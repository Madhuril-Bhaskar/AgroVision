import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os 

# Build relative path to the JSON file
file_path = os.path.join(os.path.dirname(__file__), '..', 'disease_solutions.json')

# Load the JSON data
with open(file_path, 'r') as f:
    disease_solutions = json.load(f)

# Tensorflow Model Prediction
def model_prediction(test_image):
    # Get the relative path to the model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'trained_plant_disease_model.keras')
    
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Preprocess the image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch

    # Predict
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element


#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
     # Relative path to image
    image_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'home_page.jpeg')
    st.image(image_path, use_container_width=True)
    st.markdown("""
    <style>
    h2 {
        color: #2e7d32;
        font-size: 30px;
    }
    h3 {
        color: #388e3c;
        font-size: 24px;
    }
    ul {
        line-height: 1.8;
    }
    p {
        font-size: 17px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 🌿 Welcome to the Plant Disease Recognition System

    > *"Healthy plants, healthy planet."*

    Our platform is built to empower farmers, botanists, agricultural researchers, and plant enthusiasts with the tools to **instantly identify plant diseases using advanced AI models**.

    ---

    ### 🧭 What is This System?

    This system is a **deep learning-powered web application** that detects plant diseases from leaf images. It uses a **convolutional neural network (CNN)** model trained on thousands of healthy and diseased leaf images across multiple plant species.

    Whether you're an agriculturist looking to monitor crops or a student working on a research project, this tool is designed to give you fast, accurate, and actionable insights.

    ---

    ### ⚙️ How It Works

    1. **📤 Upload Image**  
    Go to the **Disease Recognition** page and upload a clear image of a plant leaf.
    
    2. **🔬 Image Analysis**  
    Our AI model processes the image and compares it against trained plant disease classes.
    
    3. **📈 Output Result**  
    You'll instantly receive a diagnosis with the **predicted disease name** and confidence level.

    ---

    ### 🌟 Features & Benefits

    - **✅ High Accuracy**  
    Built using TensorFlow, trained on 87,000+ images across 38 classes.

    - **🖼 Real-Time Predictions**  
    Upload an image and get results in less than a second.

    - **📱 Fully Responsive UI**  
    Powered by Streamlit, works smoothly on laptops, tablets, and mobile devices.

    - **🌐 Accessible**  
    Easy-to-use interface for farmers with no technical background.

    - **📊 Scalable Design**  
    Can be expanded to include treatment recommendations and multi-language support.

    ---

    
    ### 🚀 Getting Started

    1. Click the **"Disease Recognition"** tab from the sidebar.
    2. Upload a high-quality image of a leaf.
    3. Click **Predict** and let our system do the rest!
    4. Get your result and take action immediately.
                
    ---

    <center>🌱 *Empowering farmers through AI, one leaf at a time.* 🌱</center>
    """, unsafe_allow_html=True)




#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
               
                 #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                
    
                ### 📌 Supported Crops & Diseases

                Our model currently supports crops like **Apple, Corn, Grape, Peach, Potato, Tomato, Strawberry, and more**, with diseases including:

                - Apple Scab  
                - Tomato Mosaic Virus  
                - Potato Late Blight  
                - Grape Black Rot  
                - and **30+ more**

                ---


                🌱 *Empowering farmers through AI, one leaf at a time.* 🌱
                """)

               

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_container_width=True)
    #Predict button
    if(st.button("Predict")):
        st.balloons()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))

        # Display solution from JSON
        solution = disease_solutions.get(class_name[result_index], "No treatment info available.")
        st.markdown(solution)

#st.markdown("""<hr><center>🌱 Made with ❤️ for farmers and plant lovers</center>""", unsafe_allow_html=True)

# to run type python -m streamlit run main.py