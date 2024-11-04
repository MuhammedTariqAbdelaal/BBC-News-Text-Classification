import pickle
import nltk
import gradio as gr
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download("stopwords")  # Downloads the "stopwords" resource for text preprocessing

# Load the pre-trained model and vectorizer from a pickle file
# 'model.p' is assumed to contain a dictionary with a trained model and its vectorizer
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']  # The machine learning model for text classification
vectorizer = model_dict['vectorizer']  # The TfidfVectorizer to transform input text

# Define the label mapping dictionary
# Each integer label is mapped to a specific news category
labels_dict = {0: "Business", 1: "Entertainment", 
               2: "Politics", 3: "Sports", 4: "Technology"}

# Set up NLTK components for text preprocessing
stop_words = set(stopwords.words("english"))  # Set of common English stop words
lemma = WordNetLemmatizer()  # Lemmatizer to reduce words to their base forms

# Preprocessing function
# This function tokenizes, removes stop words, and lemmatizes the input text
def preprocessing(txt):
    tokens = word_tokenize(txt)  # Split the text into words (tokens)
    # Keep only alphabetic words, remove stop words, and lemmatize the rest
    words = [lemma.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return " ".join(words)  # Join the processed words back into a single string

# Prediction function
# Takes raw input text, preprocesses it, vectorizes it, and then uses the model to predict its category
def classify_text(data):
    prep_text = preprocessing(data)  # Preprocess the input text
    vec_text = vectorizer.transform([prep_text])  # Transform the preprocessed text to the vectorized format
    pred = model.predict(vec_text)  # Predict the category using the loaded model
    predicted_category = labels_dict[int(pred[0])]  # Map the numeric prediction to the category label
    return f"Predicted category: {predicted_category}"  # Return the prediction result as a string

# Create a Gradio Interface for the classification model
iface = gr.Interface(
    fn=classify_text,  # Function to be executed for predictions
    inputs="text",  # Text input for Gradio interface
    outputs="text",  # Text output for Gradio interface
    title="BBC Text Classification",  # Title of the Gradio app
    description="<h3 style='text-align: center;'>Enter a news article, and this tool will classify it into one of five categories: Business, Entertainment, Politics, Sports, or Technology.</h3>",  # Centered description
    css=".input_textarea, .output_textarea { margin: auto; width: 70%; }"  # Center input/output boxes and adjust width
)

# Launch the Gradio web app
iface.launch()
