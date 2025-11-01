# streamlit_app.py
# -------------------------------
# Improved Email Spam Classifier
# -------------------------------

import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

ps = PorterStemmer()

def transform_text(text):
    """Cleans, tokenizes, and stems the input text."""
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    # Keep only alphanumeric tokens
    y = [i for i in text if i.isalnum()]
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    y = [i for i in y if i not in stop_words and i not in string.punctuation]
    
    # Apply stemming
    y = [ps.stem(i) for i in y]
    
    return " ".join(y)

@st.cache_resource
def load_models():
    """Loads the saved TF-IDF vectorizer and ML model from disk."""
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except FileNotFoundError:
        st.error("Model or vectorizer files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the same directory.")
        return None, None

def load_css(file_name):
    """Loads an external CSS file."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# -------------------------------
# 🎨 Page Config & Styling
# -------------------------------
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered" # Use 'centered' layout
)

# Load external CSS
load_css("style.css")

# Load models
tfidf, model = load_models()

# -------------------------------
# 🧭 Sidebar
# -------------------------------
st.sidebar.header("About This App")
st.sidebar.info(
   "This application performs spam detection by first processing email text using a Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer. This process converts the text into numerical features, which are then analyzed by an Extra Trees Classifier model to predict the final classification."
)
st.sidebar.markdown("---")
st.sidebar.caption("Made by Prashant using Streamlit")

# -------------------------------
# 🚀 App Layout
# -------------------------------

st.title("📧 Email Spam Classifier")
st.markdown(
    "Detect whether an email is **Spam** or **Not Spam** using a trained ML model."
)
st.markdown("---")

if model and tfidf:
    # --- Input Section ---
    subject = st.text_input("✉️ **Subject**", placeholder="Enter the email subject...")
    body = st.text_area("📝 **Email Body**", placeholder="Write the email content here...", height=220)

    # --- Prediction Button ---
    if st.button("🚀 Classify Email", use_container_width=True):
        if not subject.strip() and not body.strip():
            st.warning("Please enter an email subject or body to classify.")
        else:
            with st.spinner("Analyzing email..."):
                # 1. Combine and preprocess text
                combined_text = f"{subject} {body}"
                transformed_text = transform_text(combined_text)
                
                # 2. Vectorize
                vector_input = tfidf.transform([transformed_text])
                
                # 3. Predict
                result = model.predict(vector_input)[0]

                # --- Display Result ---
                st.subheader("📊 Classification Result")
                if result == 1:
                    st.error("🚫 This email is classified as **Spam**.", icon="🚫")
                else:
                    st.success("✅ This email is classified as **Not Spam**.", icon="✅")
                
                # --- Show Details ---
                with st.expander("See how the text was processed"):
                    st.code(transformed_text, language=None)
else:
    st.info("App is in setup mode. Waiting for model files...")
    
    
# # streamlit_app.py
# # -------------------------------
# # Modern Streamlit Email/SMS Spam Classifier using pre-trained pickle models
# # Includes NLTK-based preprocessing (tokenization, stopword removal, stemming)
# # -------------------------------

# import streamlit as st
# import pickle
# import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# # -------------------------------
# # 🔧 Setup & Downloads
# # -------------------------------
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# ps = PorterStemmer()

# # -------------------------------
# # 🧠 Text Preprocessing
# # -------------------------------
# def transform_text(text):
#     """Lowercase, tokenize, remove stopwords/punctuation, and apply stemming."""
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = [i for i in text if i.isalnum()]  # keep only alphanumeric tokens

#     y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

#     y = [ps.stem(i) for i in y]

#     return " ".join(y)

# # -------------------------------
# # 📦 Load Models
# # -------------------------------
# @st.cache_resource
# def load_models():
#     tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
#     model = pickle.load(open('model.pkl', 'rb'))
#     return tfidf, model

# tfidf, model = load_models()

# # -------------------------------
# # 🎨 Streamlit Page Config
# # -------------------------------
# st.set_page_config(
#     page_title="📧 Email/SMS Spam Classifier",
#     page_icon="🚀",
#     layout="centered"
# )

# # -------------------------------
# # 💅 Custom CSS
# # -------------------------------
# st.markdown("""
#     <style>
#         .main-box {
#             padding: 2rem;
#             background-color: #f9fafc;
#             border-radius: 1rem;
#             box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
#         }
#         .result-card {
#             padding: 1rem;
#             border-radius: 0.75rem;
#             text-align: center;
#             font-size: 1.2rem;
#             font-weight: 600;
#             margin-top: 1rem;
#         }
#         .spam {
#             background-color: #ffe5e5;
#             color: #b30000;
#             border: 1px solid #ffcccc;
#         }
#         .not-spam {
#             background-color: #e6ffed;
#             color: #007a33;
#             border: 1px solid #b3ffcc;
#         }
#         .footer {
#             text-align: center;
#             color: #777;
#             font-size: 0.9rem;
#             margin-top: 2rem;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # -------------------------------
# # 🧭 App Layout
# # -------------------------------
# st.title("📧 Email / SMS Spam Classifier")
# st.markdown(
#     "Classify whether a given email or SMS message is **Spam** or **Not Spam** using a trained ML model."
# )

# st.markdown('<div class="main-box">', unsafe_allow_html=True)

# # 📝 User Input
# input_sms = st.text_area("✉️ Enter the email or SMS text below:", placeholder="Type your message here...", height=180)

# # 🔍 Predict Button
# if st.button("🚀 Predict", use_container_width=True):
#     if not input_sms.strip():
#         st.warning("Please enter a message to classify.")
#     else:
#         # 1️⃣ Preprocess
#         transformed_sms = transform_text(input_sms)

#         # 2️⃣ Vectorize
#         vector_input = tfidf.transform([transformed_sms])

#         # 3️⃣ Predict
#         result = model.predict(vector_input)[0]

#         # 4️⃣ Display Result
#         st.markdown("---")
#         st.subheader("📊 Prediction Result:")

#         if result == 1:
#             st.markdown('<div class="result-card spam">🚫 This message is classified as <b>Spam</b>.</div>', unsafe_allow_html=True)
#         else:
#             st.markdown('<div class="result-card not-spam">✅ This message is classified as <b>Not Spam</b>.</div>', unsafe_allow_html=True)

# st.markdown('</div>', unsafe_allow_html=True)

# # -------------------------------
# # 🪪 Footer
# # -------------------------------
# st.markdown('<div class="footer">Built with ❤️ using Streamlit and scikit-learn</div>', unsafe_allow_html=True)
