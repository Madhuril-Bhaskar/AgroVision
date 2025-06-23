# 🌿 Plant Disease Recognition System

A deep learning-based system that detects plant leaf diseases using a trained TensorFlow model and an interactive Streamlit web app. This project uses image classification techniques to identify the disease from leaf images and suggests possible treatments.

---

## 🚀 Features

- 🌱 Identify plant leaf diseases using a trained CNN model
- 📊 Streamlit web interface for easy interaction
- 🧠 TensorFlow model trained on labeled plant disease images
- 🧾 JSON-based solution recommendations
- 📂 Structured, clean, and GitHub-ready project

---

## 🧪 Model Information

- Model Type: Convolutional Neural Network (CNN)
- Input Shape: 128x128 RGB images
- File: `trained_plant_disease_model.keras` (~90 MB)
- Framework: TensorFlow 2.17.0

---

## 🗃️ Project Structure

plant-disease-detection/
├── assets/ # Images (e.g., homepage display)
├── data/ # Folder for train/valid/test datasets (not included)
├── model/ # Trained model and training history
├── notebooks/ # Jupyter notebooks for training & testing
├── src/ # Main Streamlit app
├── disease_solutions.json # JSON mapping for disease solutions
├── requirements.txt # Python dependencies
├── .gitignore # Files/folders ignored by Git
└── README.md # You're here

---

## 📁 Dataset

The dataset used for training and testing is not included in this repository due to size limitations.

📥 **[Download Dataset][(https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset))**

After downloading, place the folders in this structure:

plant-disease-detection/
└── data/
├── train/
├── valid/
└── test/

---

## 🖥️ Installation & Running

### 1️⃣ Clone the Repository

git clone https://github.com/Madhuril-Bhaskar/Plant-Disease-Recognition.git
cd Plant-Disease-Recognition

2️⃣ Create a Virtual Environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Web App
python -m streamlit run main.py

⚙️ Requirements
Python 3.8+ and the following packages:

streamlit
tensorflow==2.17.0
numpy==1.24.3
Pillow
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.13.0
pandas==2.1.0
librosa==0.10.1

📌 Example Prediction
Upload a plant leaf image and the app will:

Display the image

Predict the disease class

Show the recommended solution from disease_solutions.json

🔒 License
This project is licensed under the MIT License — see the LICENSE file for details.

🙋‍♀️ Author
Madhuril Bhaskar
GitHub: @Madhuril-Bhaskar
