import os
import pandas as pd
import streamlit as st
import pdfplumber
import time
import base64
import tempfile

from modules.chatbot import Chatbot
from modules.embedder import Embedder
from langchain.callbacks import get_openai_callback
from langchain.chat_models import AzureChatOpenAI


class Utilities:
    # @staticmethod
    # def load_api_key():
    #      """
    #      Loads the OpenAI API key from the .env file or from the user's input
    #      and returns it
    #      """
    #      if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
    #         user_api_key = os.environ["OPENAI_API_KEY"]
    #         st.sidebar.success("API key loaded from .env", icon="🚀")
    #      else:
    #         user_api_key = st.sidebar._text_input(
    #             label="#### Your OpenAI API key 👇", placeholder="Paste your openAI API key, sk-", type="password"
    #         )
    #         if user_api_key:
    #             st.sidebar.success("API key loaded", icon="🚀")
    #      return user_api_key
    
    @staticmethod
    def handle_upload():
        """
        Handles the file upload and displays the upload file
        """
        uploaded_file = st.sidebar.file_uploader("Upload Your File Here:👇", type=["csv","pdf","txt"], label_visibility="visible")
        # Check if file was uploaded
        if uploaded_file is not None:
            uploaded_file_value = uploaded_file.getvalue()
            
            # def show_user_file(uploaded_file):
            #     file_container = st.expander("Your PDF file :")
            #     # Open PDF file
            #     with pdfplumber.open(uploaded_file) as pdf:
            #     # Loop through all pages and extract text
            #         for page_num in range(len(pdf.pages)):
            #             page = pdf.pages[page_num]
            #             text = page.extract_text()
            #             # Display text on Streamlit app
            #             file_container.write(f"Page {page_num}\n" + text)
                
            # show_user_file(uploaded_file)
            
            def show_csv_file(uploaded_file):
                file_container = st.expander("Your CSV file :")
                uploaded_file.seek(0)
                shows = pd.read_csv(uploaded_file)
                file_container.write(shows)
            
            def show_pdf_file(uploaded_file_value):
                file_container = st.expander("Your PDF file :")
                base64_pdf = base64.b64encode(uploaded_file_value).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="950" type="application/pdf"></iframe>'
                file_container.markdown(pdf_display, unsafe_allow_html=True)
            
            def show_txt_file(uploaded_file):
                file_container = st.expander("Your TXT file :")
                file_container.write(uploaded_file)
            
            def get_file_extension(uploaded_file):
                return os.path.splitext(uploaded_file)[1].lower()
            
            file_extension = get_file_extension(uploaded_file.name)
            
            # Show the contents of the file based on its extension
            if file_extension == ".csv" :
                show_csv_file(uploaded_file)
            elif file_extension == ".pdf" :
                show_pdf_file(uploaded_file_value)
            elif file_extension == ".txt" :
                show_txt_file(uploaded_file)
        else:
            st.sidebar.info(
                "👆 Upload your CSV ,PDF or TXT file to get started "
            )
            st.session_state["reset_chat"] = True
            # uploaded_file.read()
        return uploaded_file
    
    @staticmethod
    def setup_chatbot(uploaded_file, model, temperature):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", openai_api_key="e89748e90889486ab5650f74c7524a87", openai_api_base="https://insigence-azureopenai-dev-01.openai.azure.com/",openai_api_version="2023-03-15-preview", temperature=0.0, openai_api_type="azure")
        embeds = Embedder()
        with st.spinner("Processing..."):
            uploaded_file.seek(0)
            file = uploaded_file.read()
            # Get the document embeddings for the uploaded file
            vectors = embeds.getDocEmbeds(file, uploaded_file.name)
            # Create a Chatbot instance with the specified model and temperature
            chatbot = Chatbot(model, temperature, vectors)
            alert = st.success("Done...")
            time.sleep(3)
            alert.empty()
        st.session_state["ready"] = True
        return chatbot
    
    def count_tokens_agent(agent, query):
        with get_openai_callback() as cb:
            result = agent(query)
            st.write(f'Spent a total of {cb.total_tokens} tokens, costed you : {cb.total_cost}')
        return result