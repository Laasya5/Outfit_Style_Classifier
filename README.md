# 👗 Outfit Style Classifier
An AI-powered web application that classifies outfit images into categories like **Glasses/Sunglasses**, **Trousers/Jeans**, and **Shoes** using a Convolutional Neural Network (CNN) model built with TensorFlow and Flask.

## Demo
Upload an outfit image and instantly get the predicted category!

## Features

-  Outfit classification using a trained deep learning model
-  Real-time image prediction via web interface
-  Clean and intuitive UI with Flask
-  Transfer learning ready (MobileNet, ResNet compatible)
-  Trained on the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)


##  Tech Stack

| Layer         | Technology                      |
|---------------|---------------------------------|
| Frontend      | HTML, CSS, Bootstrap            |
| Backend       | Flask (Python)                  |
| ML Framework  | TensorFlow / Keras              |
| Storage       | Firebase (or local file system) |
| Dataset       | Kaggle Fashion Dataset          |



##  Dataset Info

- **Source**: [Kaggle – Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
- Preprocessing includes resizing, normalization, and class filtering.


### 1. Clone the Repository
```bash
git clone https://github.com/your-username/outfit-style-classifier.git
cd outfit-style-classifier
2. Set Up Environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Run the App
python app.py
Visit http://127.0.0.1:5001 in your browser.

📂 Project Structure

outfit-style-classifier/
│
├── app.py                 # Flask web server
├── classify.py            # Predict image using saved model
├── classification.py      # Train CNN model
├── outfit_classifier.h5   # Trained TensorFlow model
├── static/                # CSS, JS (optional)
├── templates/             # HTML templates
├── uploads/               # Uploaded images
├── requirements.txt
└── README.md

 Contributing
Contributions are welcome! Please open an issue or pull request for improvements.

 Author
Developed by Byreddy Lasya Sre Reddy & Dharani Dulla
3rd Year B.Tech 
📧 [laasyareddy394@gmail.com]
