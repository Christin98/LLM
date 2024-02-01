import os
import pickle
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Embedder:
    
    def __init__(self):
        self.PATH = "embeddings"
        self.createEmbeddingsDir()

    def createEmbeddingsDir(self):
        """
        Creates a directory to store the embeddings vectors
        """
        if not os.path.exists(self.PATH):
            os.mkdir(self.PATH)

    def storeDocEmbeds(self, file, filename):
        """
        Stores document embeddings using Langchain and FAISS
        """
        # Write the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name
            
        def get_file_extension(uploaded_file):
            file_extension =  os.path.splitext(uploaded_file)[1].lower()
            if file_extension not in [".csv", ".pdf",".txt"]:
               raise ValueError("Unsupported file type. Only CSV and PDF files are allowed.")
            return file_extension
        
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 1000,
            chunk_overlap = 50,
            length_function = len,
        )
        
        file_extension = get_file_extension(filename)

        # Load the data from the file using Langchain
        if file_extension == ".csv":
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                'delimiter': ',',
            })
            data = loader.load()
        
        elif file_extension == ".pdf":
            loader = PyPDFLoader(file_path=tmp_file_path)
            data = loader.load_and_split(text_splitter)
            # docs  = text_splitter.split_documents(data)
            # print(docs)
        elif file_extension == ".txt":
            loader = TextLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load_and_split(text_splitter)
        
        # Create an embeddings object using Langchain
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1, openai_api_key="e89748e90889486ab5650f74c7524a87")
    
        # print(embeddings)

        # Store the embeddings vectors using FAISS
        vectors = FAISS.from_documents(documents=data, embedding=embeddings)
        os.remove(tmp_file_path)

        # Save the vectors to a pickle file
        with open(f"{self.PATH}/{filename}.pkl", "wb") as f:
            pickle.dump(vectors, f)

    def getDocEmbeds(self, file, filename):
        """
        Retrieves document embeddings
        """
        # Check if embeddings vectors have already been stored in a pickle file
        if not os.path.isfile(f"{self.PATH}/{filename}.pkl"):
            # If not, store the vectors using the storeDocEmbeds function
            self.storeDocEmbeds(file, filename)

        # Load the vectors from the pickle file
        with open(f"{self.PATH}/{filename}.pkl", "rb") as f:
            vectors = pickle.load(f)

        return vectors