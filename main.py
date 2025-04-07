import streamlit as st
import recommend_app  # Import your first app
import mistral_app  # Import your second app
import deepseek_app # Import your third app

# Sidebar navigation
st.sidebar.title("🔀 Navigation")
app_selection = st.sidebar.radio("Go to", ["Product Recommendations", "Mistral", "Deepseek"])

# Load selected app

if app_selection == "Product Recommendations":
    recommend_app.run()  # Call the main function of app1.py

elif app_selection == "Mistral":
    mistral_app.run()  # Call the main function of app2.py

elif app_selection == "Deepseek":
    deepseek_app.run()  # Call the main function of app3.py
