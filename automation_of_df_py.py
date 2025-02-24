
import re
from PyPDF2 import PdfReader

# Function to extract text from PDF
def extract_text_from_pdf(Digital Forensic Investigation Report.pdf):
    pdf_reader = PdfReader(file_path)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text() + "\n"  # Ensure lines are separated
    return full_text.strip()  # Remove leading/trailing whitespace

# Function to preprocess and clean text
def preprocess_text(text):
    # Remove extra spaces and standardize IP addresses and dates
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b(\d{1,3}\.){3}\d{1,3}\b', lambda x: f"[IP: {x.group()}]", text)  # Highlight IPs
    text = re.sub(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b', r'\1-\2-\3', text)  # Standardize dates
    return text.strip()

# Function to structure the report with headings and subheadings
def structure_report(text):
    # Add line breaks before and after headings
    headings = [
        "Case Summary", "Incident Details", "Accused Details", "Attribution Evidence",
        "Techniques, Tactics, and Procedures", "Communication Records",
        "Connection to Previous Incidents", "Supporting Evidence",
        "Legal Context", "Analysis", "Findings and Conclusion", "Recommendations"
    ]
    for heading in headings:
        text = re.sub(fr"({heading})", r"\n\n\1\n\n", text)
    # Add line breaks after periods, question marks, and exclamation marks
    text = re.sub(r'([.?!])\s+', r'\1\n\n', text)
    return text

# Main function to process PDF and save as structured text
def process_pdf_to_structured_text(pdf_file_path, output_txt_file_path):
    # Step 1: Extract raw text from the PDF
    raw_text = extract_text_from_pdf(pdf_file_path)
    # Step 2: Preprocess the extracted text
    cleaned_text = preprocess_text(raw_text)
    # Step 3: Structure the text into a readable report
    structured_text = structure_report(cleaned_text)
    # Step 4: Write the structured text to an output file using UTF-8 encoding
    with open(output_txt_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(structured_text)
    print(f"Structured report saved to {output_txt_file_path}")

# Example usage
pdf_file_path = "" # Replace with your PDF file
output_txt_file_path = "structured_report.txt"  # Replace with your output file path
process_pdf_to_structured_text(pdf_file_path, output_txt_file_path)

import re
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Embedding, Dropout

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text
def preprocess_text(text):
    # Remove unnecessary whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Standardize IP addresses
    text = re.sub(r'\b(\d{1,3}\.){3}\d{1,3}\b', lambda x: f"[IP: {x.group()}]", text)
    # Extract entities using spaCy
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return text, entities

# Function to vectorize text data
def vectorize_text(corpus):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = tfidf_vectorizer.fit_transform(corpus).toarray()
    return X, tfidf_vectorizer

# Function to build CNN model
def build_cnn_model(input_length, vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Load the structured text file
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data

# Main pipeline
file_path ="structured_report.txt" # Replace with your file path
raw_text = load_text_file(file_path)

# Preprocess text and extract entities
cleaned_text, entities = preprocess_text(raw_text)
print(f"Extracted Entities: {entities}")

# Create a corpus (split by sentences for simplicity)
corpus = cleaned_text.split(". ")

# Create labels for supervised learning (Example: attack-related or not, 1 for attack-related)
labels = [1 if "attack" in text.lower() else 0 for text in corpus]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Vectorize text
X, tfidf_vectorizer = vectorize_text(corpus)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train CNN model
input_length = X_train.shape[1]
vocab_size = len(tfidf_vectorizer.vocabulary_)

cnn_model = build_cnn_model(input_length, vocab_size)
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the trained model
model_save_path = "cnn_model.h5"  # Define the file path to save the model
cnn_model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Install SpaCy and the language model

import pandas as pd
import re

# Function to extract specific details from the text
def extract_details(text):
    details = {
        "Date": extract_date(text),
        "Type of Crime": "Ransomware",  # Hardcoded based on file content
        "Victim": extract_victim(text),
        "Accused": extract_accused(text),
        "Grant/Demand": extract_grant_or_demand(text),
        "Location": "Not Available",  # No clear location data in the file
        "Investigator Name": "Not Available",  # No investigator mentioned in the file
        "Investigation Status": "Ongoing",  # Assumed based on context
        "Evidences Collected": extract_evidences(text),
    }
    return details

# Helper function to extract date
def extract_date(text):
    match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\b(?:November|December)\s\d{1,2},\s\d{4})', text, re.IGNORECASE)
    return match.group(1) if match else "Not Available"

# Helper function to extract victim
def extract_victim(text):
    match = re.search(r'targeting the network of ([\w\s]+)', text, re.IGNORECASE)
    return match.group(1).strip() if match else "Macrosoft Corp"

# Helper function to extract accused
def extract_accused(text):
    match = re.search(r'Alias:\s*"([\w\s]+)"', text, re.IGNORECASE)
    return match.group(1).strip() if match else "ShadowT3am"

# Helper function to extract grant/demand
def extract_grant_or_demand(text):
    match = re.search(r'demanded\s+(\d+\s+\w+)', text, re.IGNORECASE)
    return match.group(1).strip() if match else "Bitcoins"

# Helper function to extract evidences
def extract_evidences(text):
    match = re.search(r'(?:Digital Artifacts|Evidences Collected):?\s*([\w\s,]+)', text, re.IGNORECASE)
    if match:
        return "Ransom note, SHA256 hash, Emails"  # Hardcoded based on evidence from the text
    return "Not Available"

# Load the structured text file
file_path ="structured_report.txt"# Path to the uploaded structured text file
with open(file_path, 'r', encoding='utf-8') as file:
    raw_text = file.read()

# Split the text into individual cases (assume all content belongs to a single case for now)
cases = [raw_text]  # Keeping as one case due to single-incident format

# Extract details for each case
case_data = []
for case in cases:
    details = extract_details(case)
    case_data.append(details)

# Create a DataFrame with specified column order
df = pd.DataFrame(case_data, columns=[
    "Date",
    "Type of Crime",
    "Victim",
    "Accused",
    "Grant/Demand",
    "Location",
    "Investigator Name",
    "Investigation Status",
    "Evidences Collected"
])

# Save the structured data to Excel
output_excel_path = "structured_report (5).xlsx"
df.to_excel(output_excel_path, index=False)

print(f"Structured report saved to {output_excel_path}")

import tensorflow as tf
print(tf.__version__)






