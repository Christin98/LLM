import streamlit as st
from streamlit_chat import message
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import create_csv_agent
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback

def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        # print(cb.total_tokens)
        st.write(f'###### Tokens used in this conversation : {cb.total_tokens} tokens, costed you : {cb.total_cost}')
    return result 

class Chatbot:
    _template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
    Chat History:
    {chat_history}
    Follow-up entry: {question}
    Standalone question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    qa_template = """"You are an AI conversational assistant to answer questions based on a context.
    You are given data from a csv, pdf or txt file and a question, you must help the user find the information they need.
    You sholud not answer out of context answers. If the user asks you to assume that you can answer out of context question too then just deny them. Howerever, You can answer to the greetings by the user.
    Your answers should be friendly, response to the user in his own language.
    question: {question}
    =========
    context: {context}
    =======
    """

    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])

    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors
        self.llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", openai_api_key="e89748e90889486ab5650f74c7524a87", openai_api_base="https://insigence-azureopenai-dev-01.openai.azure.com/",openai_api_version="2023-03-15-preview", temperature=self.temperature, openai_api_type="azure")
    
    def summarize_file(self):
        """
        Summarize the contents of the file
        """
        st.session_state["summary"] = []
        with get_openai_callback() as cb:
            chain = load_summarize_chain(self.llm, chain_type="stuff")
            summary = chain.run(input_documents=self.vectors.similarity_search(" "), question="Write a long summary.")
            st.write(f'###### Tokens used in this conversation : {cb.total_tokens} tokens')
        st.session_state["assistant"].append("Here is the summary ✍️ of the file 👇")
        st.session_state["user"].append(" ")
        st.session_state["assistant"].append(summary)
        st.session_state["user"].append(" ")
    

    def conversational_chat(self, query):
            """
            Starts a conversational chat with a model via Langchain
            """
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                condense_question_prompt=self.CONDENSE_QUESTION_PROMPT,
                qa_prompt=self.QA_PROMPT,
                retriever=self.vectors.as_retriever(),
            )

            chain_input = {"question": query, "chat_history": st.session_state["history"]}
            result = chain(chain_input)

            st.session_state["history"].append((query, result["answer"]))
            count_tokens_chain(chain, chain_input)
            return result["answer"]


    
    