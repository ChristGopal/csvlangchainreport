import streamlit as st
import pandas as pd
# from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import create_pandas_dataframe_agent

from langchain.llms import OpenAI
import os

# Load OpenAI API key from environment variables or directly set here
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.warning("Please set the OpenAI API key.")
    st.stop()

# Initialize OpenAI model with LangChain
llm = OpenAI(api_key=openai_api_key, temperature=0.0)

# Streamlit UI
st.title("Analyze Any CSV File with LangChain and OpenAI")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("### CSV Data Preview")
    st.dataframe(df.head())  # Show the first few rows of the CSV

    # Create a LangChain agent for DataFrame interaction
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    # Ask a question about the data
    question = st.text_input("Ask a question about your data:")
    if question:
        with st.spinner("Analyzing your data..."):
            try:
                # Get answer from the LangChain agent
                answer = agent.run(question)
                st.write("### Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
