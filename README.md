# 🎥 Violence Detection using open camera as well as in Videos 

This project is a Flask web application that allows users to upload videos and detect violence in them using a pre-trained Keras model. To enable real-time violence detection, the system incorporates streaming capabilities, allowing it to analyze live video feeds for immediate detection of violent activities. Upon detecting violence, the system triggers a notification mechanism to alert authorities or relevant stakeholders, facilitating quick responses to potential threats.

## ✨ Features

- 📁 **Upload Videos**: Users can upload video files to be analyzed.
- 🎞️ **Live Preview**: Stream the uploaded video and view the analysis results in real-time.
- 🔍 **Violence Detection**: Uses a pre-trained Keras model to detect violence in video frames.
- ✉️ **Email Notifications**: Sends an email notification with the detected frame if violence is detected.
- 📷 **Webcam Feed**: Option to analyze live feed from the webcam.

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- Flask
- OpenCV
- Keras
- NumPy
- smtplib
- requests

## 🚀 Clone the Repository

```bash
git clone https://github.com/dinesh-fullstackwebdeveloper/violence-detection-using-flask-web-application.git
cd violence-detection-using-flask-web-application
```
## 📦 Create a Virtual Environment and Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```
## 🔧 Set Environment Variables

Create a '.env' file in the root directory of your project and add the following environment variables:

```makefile
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_email_password
EMAIL_RECIPIENT=recipient_email@gmail.com
```
## ▶️ Run the Application

```bash
python app.py
```
The application will be available at 'http://127.0.0.1:5000/'.

## 📖 Usage

### 📤 Upload a Video

Navigate to 'http://127.0.0.1:5000/'.
Click on the "Choose File" button and select a video file.
Click on the "Upload" button.

### 📺 View and Analyze the Video

After uploading, the video file name will be displayed.
Click on the "Preview" link to view the video.
The application will start processing the video, and the results will be displayed in real-time.

### 📧 Receive Email Notifications

If violence is detected in the video, an email notification will be sent to the specified recipient.

## 📁 Project Structure

```bash
violence-detection/
│
├── static/
│   └── uploads/          # Directory for uploaded video files
│
├── templates/
│   ├── index.html        # Upload page template
│   ├── preview.html      # Preview page template
│
├── app.py                # Main application file
├── requirements.txt      # Required Python packages
├── README.md             # Project README file
└── .env                  # Environment variables
```
## 🧠 Model Information

The pre-trained model used in this project is a Keras model trained to detect violence in video frames. The model file ('Own_dataset_mobi_Lstm3.h5') should be placed in the root directory of the project.

