import streamlit as st
import joblib

# --- 1. LOAD THE SAVED MODEL AND VECTORIZER ---
# We load the objects we saved in the 'train_model.py' script
try:
    model = joblib.load('complaint_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    st.error("Error: Model or vectorizer files not found.")
    st.stop() # Stop the app if files aren't found
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    st.stop()


# --- 2. CREATE THE WEB APP INTERFACE ---

# Set the title of the web app
st.title('Complaint Analyser 🤖')

# Add a description
st.markdown("""
This web app uses a Machine Learning model to analyse customer complaints 
and predict the product category.
""")
st.markdown("---") # Adds a horizontal line

# Create a text area for user input
st.subheader("Enter Your Complaint Text:")
user_input = st.text_area("Type or paste your complaint here...", height=150)

# Create a button to trigger the analysis
if st.button('Analyse Complaint'):
    if user_input:
        # --- 3. MAKE PREDICTION ---
        
        # Transform the user input using the loaded vectorizer
        # We put [user_input] in a list because the vectorizer expects a list of texts
        input_vector = vectorizer.transform([user_input])
        
        # Make a prediction using the loaded model
        prediction = model.predict(input_vector)
        
        # Get the probability scores (optional but cool)
        probabilities = model.predict_proba(input_vector)
        max_probability = probabilities.max()
        
        # --- 4. DISPLAY THE RESULT ---
        st.subheader("Analysis Result:")
        st.success(f"Predicted Category: {prediction[0]}")
        st.info(f"Confidence Score: {max_probability * 100:.2f}%")
        
    else:
        # Show a warning if the user clicks the button with no text
        st.warning('Please enter a complaint to analyse.')