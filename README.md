# CNC Milling Performance Analysis and Fault Detection

# 📌 Overview
This project focuses on analyzing CNC milling machine performance and detecting faults using deep learning techniques. The primary objective is to predict tool wear condition, machining finalization status, and passed visual inspection using time-series sensor data collected from CNC milling experiments. The project includes data collection, preprocessing, deep learning model training (LSTM), and deployment via Streamlit.

# 📂 Project Structure
CNC-Milling-Performance-Analysis-and-Fault-Detection/
│
├── Data Collection & Preprocessing
│   ├── Merging experiment files
│   ├── Handling missing values
│   ├── Encoding categorical features
│   ├── Standardizing numerical features
│
├── Model Development
│   ├── LSTM-based deep learning model
│   ├── Multi-output classification
│   ├── Model training & evaluation
│   ├── Saving trained model & preprocessing objects
│
├── Deployment
│   ├── Streamlit-based web application
│   ├── Background image integration
│   ├── File upload functionality
│   ├── Model inference & results visualization

# 📊 Data Processing
Data Source: CNC milling sensor data from multiple experiment files.
Preprocessing Steps:
Merging experiment files with train data
Handling missing values (replacing with default values)
Encoding categorical features (Label Encoding)
Standardizing numerical features (using StandardScaler)
Reshaping data for LSTM (time-series input format)

# 🏗 Model Architecture
Model Type: LSTM-based deep learning model
Input Shape: (timesteps=1, features=47)
Hidden Layers:
2 LSTM layers (50 neurons each, dropout layers for regularization)
Fully connected dense layers (ReLU activation)
Output Layers:
tool_condition (softmax activation, 2 classes: unworn/worn)
machining_finalized (softmax activation, 2 classes: yes/no)
passed_visual_inspection (softmax activation, 2 classes: yes/no)
Loss Function: Sparse Categorical Crossentropy
Optimizer: Adam
# 🚀 Deployment with Streamlit
Frontend: User-friendly web interface built with Streamlit
Features:
CSV file upload
Preprocessing & model inference
Display predictions in tabular format
Downloadable results in CSV format
Background image customization
Live Demo: Streamlit App

# 📜 Requirements
streamlit
pandas
numpy
tensorflow
scikit-learn
joblib
pickle-mixin

# 📥 Installation & Usage
Clone the repository:
git clone https://github.com/harivign/CNC-Milling-Performance-Analysis-and-Fault-Detection.git
Install dependencies:
pip install -r requirements.txt
Run the Streamlit app:
streamlit run streamlit_app.py
Upload a CSV file and get predictions!

# 🎯 Key Takeaways
Successfully implemented an LSTM model for multi-output classification.
Integrated preprocessing, training, and deployment workflows.
Built an interactive Streamlit web app for real-time predictions.
Enhanced visualization with background customization.

# 🏆 Achievements
✅ Merged and processed raw CNC sensor data.
✅ Trained an LSTM model for fault detection.
✅ Deployed a user-friendly tool wear prediction app.
✅ Hosted the project on GitHub & Streamlit.

# 🔗 Links
📂 GitHub Repository: CNC-Milling-Performance-Analysis-and-Fault-Detection
🌐 Streamlit Web App: Live Demo
