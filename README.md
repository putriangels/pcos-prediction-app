# 🩺 PCOS Prediction using CNN on Ultrasound Images

[![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-%23FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)](https://www.python.org/)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

This application uses **Convolutional Neural Networks (CNN)** to predict **Polycystic Ovary Syndrome (PCOS)** from ovary ultrasound images. Built with **Streamlit**, it allows healthcare professionals and researchers to analyze images quickly and accurately.

---

## 💡 Features

- ✅ Upload ultrasound images and receive prediction (Positive/Negative PCOS)
- ✅ Select CNN architecture: DenseNet201, VGG19, or InceptionV3
- ✅ Select optimizer: Adam or SGD
- ✅ Select learning rate: 0.001 or 0.0001
- ✅ View evaluation metrics: accuracy plot, loss plot, confusion matrix, and classification report

---

## 🛠 Technologies

- Python
- TensorFlow
- Streamlit
- Matplotlib
- Seaborn
- NumPy
- Pandas
- Scikit-learn

---

## 🚀 How to Run

```bash
# 1. Clone this repo
git clone https://github.com/putriangels/pcos-cnn-app.git
cd pcos-cnn-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run main.py
```
