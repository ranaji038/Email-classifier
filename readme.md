# 📧 Email Classifier

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Version](https://img.shields.io/badge/version-1.0.0-blue)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-yellow)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

> A lightweight **Email Classification System** that uses Natural Language Processing (NLP) and Machine Learning to predict whether an email is spam or not.

---

## 🧠 What the Project Does

**Email Classifier** is an end-to-end text classification project that analyzes email text (subject and body) to determine its category (e.g., _Spam_ or _Not Spam_).  
It includes:

- Preprocessing using **NLTK** (tokenization, stopword removal, stemming)
- Feature extraction using **TF-IDF vectorization**
- Model training with **scikit-learn**
- Interactive UI built with **Streamlit**
- Ready-to-use **Pickle (.pkl)** models for deployment

---

## 💡 Why the Project Is Useful

This project demonstrates how to build and deploy a machine learning model for real-world email filtering tasks.  
Key features include:

✅ **Automated text cleaning and preprocessing**  
✅ **Spam detection with high accuracy** using trained ML models  
✅ **Interactive Streamlit app** for classifying email subject + body  
✅ **Easily extendable** — retrain with your own data or models  
✅ **Portable & lightweight** — deploy anywhere with minimal setup

---

## ⚙️ How to Get Started

### 🧩 Prerequisites

Make sure you have the following installed:

- Python ≥ 3.8
- pip (Python package manager)

Optional but recommended:

- Virtual environment (`venv` or `conda`)

---

### 📦 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ranaji038/Email-classifier.git
   cd Email-classifier
   ```

````

2. **Create and activate a virtual environment (optional)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # For Linux/Mac
   .venv\Scripts\activate       # For Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

### 🚀 Run the Streamlit App

To launch the web interface for classification:

```bash
streamlit run streamlit_app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501)

---

### 🧰 Usage Example

You can also use the model directly in Python:

```python
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load pre-trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [ps.stem(i) for i in text if i.isalnum() and i not in stopwords.words('english')]
    return " ".join(y)

# Example email
subject = "Congratulations! You've won a $500 gift card!"
body = "Click here to claim your reward. This offer expires soon."
combined_text = subject + " " + body

# Transform & predict
input_vector = tfidf.transform([transform_text(combined_text)])
prediction = model.predict(input_vector)[0]

print("Prediction:", "Spam" if prediction == 1 else "Not Spam")
```

---

## 🧭 Project Structure

```
Email-classifier/
│
├── streamlit_app.py           # Streamlit UI
├── model.pkl                  # Trained ML model
├── vectorizer.pkl             # TF-IDF vectorizer
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── data/                      # (optional) Training data
└── docs/                      # Additional documentation (optional)
```

---

## 🧪 Model Details

- **Preprocessing:** NLTK (tokenization, stemming, stopword removal)
- **Feature extraction:** TF-IDF Vectorization
- **Model:** ExtraTreesClassifier (can be replaced with SVM, NB, etc.)
- **Evaluation:** Accuracy, precision, recall (available via `train_model.py`)

---

## 🛠️ Development

To retrain the model:

```bash
python train_model.py
```

You can modify the dataset or preprocessing pipeline in the script before retraining.

---

## 🆘 Where to Get Help

If you run into any issues:

- Open an issue under the [Issues](../../issues) tab
- Check out the `docs/` folder (if available)
- Reach out to the maintainer directly

---

## 👨‍💻 Who Maintains & Contributes

**Maintainer:**
**Prashant Singh Ranawat**
Backend Developer | REST API & Cloud Integration | Bengaluru, India

Contributions are welcome! 🙌

### 🧩 How to Contribute

1. Fork the repository
2. Create a new feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Commit and push your changes
4. Open a Pull Request

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for full guidelines.

---

## 📄 License

This project is licensed under the terms of the **MIT License**.
See the [LICENSE](LICENSE) file for details.

---

## ⭐ Acknowledgments

- [NLTK](https://www.nltk.org/) for text preprocessing
- [Scikit-learn](https://scikit-learn.org/) for model training
- [Streamlit](https://streamlit.io/) for building the UI

---

### 🏁 Quick Links

- [🧾 Issues](../../issues)
- [📜 LICENSE](LICENSE)
- [🧠 Contributing Guidelines](docs/CONTRIBUTING.md)
- [🚀 Demo App Code](streamlit_app.py)

---

> _Built with ❤️ by [Prashant Singh Ranawat](https://github.com/ranaji038)_
> Empowering text intelligence, one email at a time.

```

```
````
