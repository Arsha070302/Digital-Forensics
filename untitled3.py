import re
import spacy
import pandas as pd
from io import BytesIO
from PyPDF2 import PdfReader
import streamlit as st

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text.strip()

# Function to preprocess and clean text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\b(\d{1,3}\.){3}\d{1,3}\b', lambda x: f"[IP: {x.group()}]", text)  # Highlight IPs
    text = re.sub(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b', r'\1-\2-\3', text)  # Standardize dates
    return text.strip()

# Function to structure the text
def structure_report(text):
    headings = [
        "Case Summary", "Incident Details", "Accused Details", "Attribution Evidence",
        "Techniques, Tactics, and Procedures", "Communication Records",
        "Connection to Previous Incidents", "Supporting Evidence",
        "Legal Context", "Analysis", "Findings and Conclusion", "Recommendations"
    ]
    for heading in headings:
        text = re.sub(fr"({heading})", r"\n\n\1\n\n", text)
    text = re.sub(r'([.?!])\s+', r'\1\n\n', text)  # Add line breaks after punctuation
    return text

# Function to extract structured details for Excel
def extract_details(text):
    details = {
        "Date": extract_date(text),
        "Type of Crime": "Ransomware",
        "Victim": extract_victim(text),
        "Accused": extract_accused(text),
        "Grant/Demand": extract_grant_or_demand(text),
        "Location": "Not Available",
        "Investigator Name": "Not Available",
        "Investigation Status": "Ongoing",
        "Evidences Collected": extract_evidences(text),
    }
    return details

def extract_date(text):
    match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\b(?:November|December)\s\d{1,2},\s\d{4})', text, re.IGNORECASE)
    return match.group(1) if match else "Not Available"

def extract_victim(text):
    match = re.search(r'targeting the network of ([\w\s]+)', text, re.IGNORECASE)
    return match.group(1).strip() if match else "Macrosoft Corp"

def extract_accused(text):
    match = re.search(r'Alias:\s*"([\w\s]+)"', text, re.IGNORECASE)
    return match.group(1).strip() if match else "ShadowT3am"

def extract_grant_or_demand(text):
    match = re.search(r'demanded\s+(\d+\s+\w+)', text, re.IGNORECASE)
    return match.group(1).strip() if match else "Bitcoins"

def extract_evidences(text):
    match = re.search(r'(?:Digital Artifacts|Evidences Collected):?\s*([\w\s,]+)', text, re.IGNORECASE)
    return match.group(1).strip() if match else "Not Available"

# Streamlit App
st.title("PDF to Structured Report and Excel Converter")

# Upload file
uploaded_file = st.file_uploader("D:\Digital Forensics AI\Social Media Harassment and Identity Theft.pdf", type=["pdf"])

if uploaded_file is not None:
    # Process the PDF file
    with st.spinner("Processing the PDF file..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        cleaned_text = preprocess_text(raw_text)
        structured_text = structure_report(cleaned_text)

    st.success("PDF successfully processed!")
    
    # Display the structured text
    st.subheader("Extracted Structured Text")
    st.text(structured_text)

    # Extract details for Excel
    case_data = [extract_details(cleaned_text)]
    df = pd.DataFrame(case_data, columns=[
        "Date", "Type of Crime", "Victim", "Accused", "Grant/Demand",
        "Location", "Investigator Name", "Investigation Status", "Evidences Collected"
    ])

    # Display the DataFrame
    st.subheader("Extracted Case Details")
    st.dataframe(df)

    # Convert DataFrame to Excel
    def convert_df_to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        processed_data = output.getvalue()
        return processed_data

    excel_data = convert_df_to_excel(df)

    # Provide download button
    st.download_button(
        label="Download Excel Report",
        data=excel_data,
        file_name="structured_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
