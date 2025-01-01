import streamlit as st
import pandas as pd
import numpy as np
import nltk
import pickle
import re

nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model
with open('best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Load the TF-IDF vectorizer - make sure you have the correct file name
with open('tfidf_vectorizer.pkl', 'rb') as file:  # Change this to your vectorizer file name
    tf = pickle.load(file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:  # Change this to your label encoder file name
    le = pickle.load(file)

def cleanResume(txt):
    cleantxt = re.sub('https\S+\s', " ", txt)  # remove https
    cleantxt = re.sub('@\S+', " ", cleantxt)   # remove @
    cleantxt = re.sub('#\S+', " ", cleantxt)   # remove #
    cleantxt = re.sub('RT|cc', " ", cleantxt)  # remove rt, cc
    cleantxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), " ", cleantxt)
    cleantxt = re.sub('\s+', " ", cleantxt)
    cleantxt = re.sub('\n', " ", cleantxt)
    cleantxt = re.sub(r'[^\x00-\x7f]', " ", cleantxt)
    return cleantxt.lower().strip()  # Added lower() and strip() for better preprocessing

def main():
    st.title('Resume Screening Web App')
    st.write('Upload a resume to predict its category')
    
    file_upload = st.file_uploader('Upload resume', type=['pdf', 'txt', 'docx'])

    if file_upload is not None:
        try:
            # Show file details
            st.write("Filename:", file_upload.name)
            
            # Read and decode file
            try:
                resume_bytes = file_upload.read()
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                resume_text = resume_bytes.decode('latin-1')
            
            # Clean the resume text
            cleaned_resume = cleanResume(resume_text)
            
            # Show the cleaned text (optional, for debugging)
            with st.expander("View Processed Text"):
                st.write(cleaned_resume[:1000] + "..." if len(cleaned_resume) > 1000 else cleaned_resume)
            
            if st.button('Predict Category'):
                # Transform the text using TF-IDF
                resume_vector = tf.transform([cleaned_resume])
                
                # Make prediction
                prediction = best_model.predict(resume_vector)[0]
                
                # Convert prediction to category name
                category = le.inverse_transform([prediction])[0]
                
                # Display result
                st.success(f'Predicted Category: {category}')
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please make sure you upload a valid file")

if __name__ == '__main__':
    main()