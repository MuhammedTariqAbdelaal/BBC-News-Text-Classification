import pickle
import nltk
import gradio as gr
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download("stopwords")

# Load the model and vectorizer
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
vectorizer = model_dict['vectorizer']

# Define labels
labels_dict = {0: "Business", 1: "Entertainment", 
               2: "Politics", 3: "Sports", 4: "Technology"}

# Set up NLTK components
stop_words = set(stopwords.words("english"))
lemma = WordNetLemmatizer()

# Preprocessing function
def preprocessing(txt):
    tokens = word_tokenize(txt)
    words = [lemma.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return " ".join(words)

# Prediction function
def classify_text(data):
    prep_text = preprocessing(data)
    vec_text = vectorizer.transform([prep_text])
    pred = model.predict(vec_text)
    predicted_category = labels_dict[int(pred[0])]
    return f"Predicted category: {predicted_category}"

# Create Gradio Interface
iface = gr.Interface(
    fn=classify_text,
    inputs="text",
    outputs="text",
    title="BBC Text Classification",
    description="<h3 style='text-align: center;'>Enter a news article, and this tool will classify it into one of five categories: Business, Entertainment, Politics, Sports, or Technology.</h3>",
    css=".input_textarea, .output_textarea { margin: auto; width: 70%; }"  # Center input/output boxes
)

# Launch the app
iface.launch()
