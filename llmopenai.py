import os
import sys
import re
import streamlit as st
from io import BytesIO
from io import StringIO
from dotenv import load_dotenv
from langchain.agents import create_csv_agent
# from streamlit_chat import message
# from langchain.embeddings.openai import OpenAIEmbeddings
# # from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import PromptTemplate
# # from langchain.document_loaders.csv_loader import CSVLoader
# from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
# from langchain.schema import HumanMessage
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import TokenTextSplitter
from typing import List
# import pdfplumber
import openai
from modules.history import ChatHistory
from modules.layout import Layout
from modules.utils import Utilities
from modules.sidebar import Sidebar
from modules.chatbot import Chatbot
# import tempfile

def reload_module(module_name):
    import importlib
    import sys
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return sys.modules[module_name]

chatbot_module = reload_module('modules.chatbot')
history_module = reload_module('modules.history')
layout_module = reload_module('modules.layout')
utils_module = reload_module('modules.utils')
sidebar_module = reload_module('modules.sidebar')

Chatbot = chatbot_module.Chatbot
ChatHistory = history_module.ChatHistory
Layout = layout_module.Layout
Utilities = utils_module.Utilities
Sidebar = sidebar_module.Sidebar

def init():
    load_dotenv()
    st.set_page_config(layout="wide", page_icon="💬", page_title="FileGenie")

def main():
    
    init()
    openai.api_type = "azure"
    openai.api_base = os.getenv('OPENAI_API_BASE')
    openai.api_key = os.getenv("OPENAI_API_KEY")
    load_dotenv()
    layout, sidebar, utils = Layout(), Sidebar(), Utilities()
    layout.show_header()

    
    uploaded_file = utils.handle_upload()
    
    if uploaded_file:
        history = ChatHistory()
        sidebar.show_options()
        try:
            chatbot = utils.setup_chatbot(
                uploaded_file, st.session_state["model"], st.session_state["temperature"]
            )
            st.session_state["chatbot"] = chatbot
            
            if st.session_state["ready"]:
                response_container, prompt_container = st.container(), st.container()
                
                with prompt_container:
                    is_ready, user_input = layout.prompt_form()
                    
                    history.initialize(uploaded_file)
                    if st.session_state["reset_chat"]:
                        history.reset(uploaded_file)
                        st.session_state["chatbot"].summarize_file()

                    if is_ready:
                        history.append("user", user_input)
                        output = st.session_state["chatbot"].conversational_chat(user_input)
                        history.append("assistant", output)
                history.generate_messages(response_container)
                
                if st.session_state["show_csv_agent"]:

                    query = st.text_input(label="Use CSV agent for precise information about the structure of your csv file", 
                                            placeholder="e-g : how many rows in my file ?"
                                            )
                    if query != "":

                        # format the CSV file for the agent
                        uploaded_file_content = BytesIO(uploaded_file.getvalue())

                        old_stdout = sys.stdout
                        sys.stdout = captured_output = StringIO()

                        # Create and run the CSV agent with the user's query
                        agent = create_csv_agent(AzureChatOpenAI(temperature=0, deployment_name="gpt-35-turbo", openai_api_key="e89748e90889486ab5650f74c7524a87", openai_api_base="https://insigence-azureopenai-dev-01.openai.azure.com/",openai_api_version="2023-03-15-preview", openai_api_type="azure"), uploaded_file_content, verbose=True, max_iterations=4)
                        result = agent.run(query)

                        sys.stdout = old_stdout

                        # Clean up the agent's thoughts to remove unwanted characters
                        thoughts = captured_output.getvalue()
                        cleaned_thoughts = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', thoughts)
                        cleaned_thoughts = re.sub(r'\[1m>', '', cleaned_thoughts)

                        # Display the agent's thoughts
                        with st.expander("Display the agent's thoughts"):
                            st.write(cleaned_thoughts)
                            Utilities.count_tokens_agent(agent, query)

                        st.write(result)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            
if __name__ == "__main__":
    main()
    
# uploaded_file = st.sidebar.file_uploader("Upload Your Pdf File Here:👇", type="pdf", label_visibility="visible")

# # Check if file was uploaded
# if uploaded_file is not None:
#     # Open PDF file
#     with pdfplumber.open(uploaded_file) as pdf:
#         # Loop through all pages and extract text
#         for page_num in range(len(pdf.pages)):
#             page = pdf.pages[page_num]
#             text = page.extract_text()
#             # Display text on Streamlit app
#             st.write(f"Page {page_num}")
#             st.write(text)

# # class NewAzureOpenAI(AzureOpenAI):
# #     stop: List[str] = None
# #     @property
# #     def _invocation_params(self):
# #         params = super()._invocation_params
# #         # fix InvalidRequestError: logprobs, best_of and echo parameters are not available on gpt-35-turbo model.
# #         params.pop('logprobs', None)
# #         params.pop('best_of', None)
# #         params.pop('echo', None)
# #         params['stop'] = self.stop
# #         return params

# # user_api_key = st.sidebar.text_input(
# #     label="#### Your OpenAI API key 👇",
# #     placeholder="Paste your openAI API key, sk-",
# #     type="password")

# # Azure Open AI Cred
# # os.environ['OPEN_API_VERSION'] = "2023-03-15-preview"
# openai.api_base = os.getenv('OPENAI_API_BASE')
# openai.api_key = os.getenv("OPENAI_API_KEY")
# # openai.api_type = "azure"
# # openai.api_base = "https://insigence-azureopenai-dev-01.openai.azure.com/"
# # openai.api_version = "2022-12-01"
# # openai.api_key = "9d9e56201eb34311b927d690c5ba2989"

# # llm = NewAzureOpenAI(deployment_name="openai",model_name='gpt-35-turbo')

# # print(llm("What is the capital of Italy?"))

# llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", openai_api_key="e89748e90889486ab5650f74c7524a87", openai_api_base="https://insigence-azureopenai-dev-01.openai.azure.com/",openai_api_version="2023-03-15-preview")
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)

# # print(llm([HumanMessage(content="What is the capital of India?")]).content)

# loader = TextLoader('Dolly.txt', 'utf-8')

# documents = loader.load()
# text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# db = FAISS.from_documents(documents=docs, embedding=embeddings)

# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:""")

# qa = ConversationalRetrievalChain.from_llm(llm=llm,
#                                            retriever=db.as_retriever(),
#                                            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
#                                            return_source_documents=True,
#                                            verbose=False)
# chat_history = []
# query = "what is Dolly Service?"
# result = qa({"question": query, "chat_history": chat_history})

# print("Question:", query)
# print("Answer:", result["answer"])

# chat_history = [(query, result["answer"])]
# query = "Which regions does the service support?"
# result = qa({"question": query, "chat_history": chat_history})

# print("Question:", query)
# print("Answer:", result["answer"])



# # uploaded_file = st.sidebar.file_uploader("upload", type="csv")

# # if uploaded_file :
# #    #use tempfile because CSVLoader only accepts a file_path
# #     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
# #         tmp_file.write(uploaded_file.getvalue())
# #         tmp_file_path = tmp_file.name

# #     loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
# #     data = loader.load()
# #     # os.environ["OPENAI_API_KEY"] = openai.api_key
# #     # loader = CSVLoader(file_path=r"C:\Users\mande\Downloads\Divi-Engine-WooCommerce-Sample-Products.csv", encoding="utf-8")
# #     # data = loader.load()

# #     embeddings = OpenAIEmbeddings()
# #     vectorstore = FAISS.from_documents(data, embeddings)

# #     chain = ConversationalRetrievalChain.from_llm(llm = AzureOpenAI(deployment_name="openai",model_name='gpt-35-turbo'),
# #                                                                       retriever=vectorstore.as_retriever())
    
# #     def conversational_chat(query):
# #         result = chain({"question": query, "chat_history": st.session_state['history']})
# #         st.session_state['history'].append((query, result["answer"]))
        
# #         return result["answer"]
    
# #     if 'history' not in st.session_state:
# #         st.session_state['history'] = []

# #     if 'generated' not in st.session_state:
# #         st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

# #     if 'past' not in st.session_state:
# #         st.session_state['past'] = ["Hey ! 👋"]
    
# #     # chat_history = []
# #     # query = "Give number items with name Divi Engine"
# #     # response = chain({"question": query , "chat_history": chat_history})

# #     # print(response["answer"])
    
# #      #container for the chat history
# #     response_container = st.container()
# #     #container for the user's text input
# #     container = st.container()

# #     with container:
# #         with st.form(key='my_form', clear_on_submit=True):
            
# #             user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
# #             submit_button = st.form_submit_button(label='Send')
            
# #         if submit_button and user_input:
# #             output = conversational_chat(user_input)
            
# #             st.session_state['past'].append(user_input)
# #             st.session_state['generated'].append(output)

# #     if st.session_state['generated']:
# #         with response_container:
# #             for i in range(len(st.session_state['generated'])):
# #                 message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
# #                 message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

