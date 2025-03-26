
import os
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)

# Set up ChatGroq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load PDF Document
def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        print(f"Loaded {len(pages)} pages from PDF.")
        if len(pages) == 0:
            raise ValueError("No pages loaded from PDF")
        return pages
    except Exception as e:
        print(f"Error loading PDF: {str(e)}")
        raise

# Split text into chunks
def split_text(pages, chunk_size=1000, chunk_overlap=200):
    if not pages:
        raise ValueError("No pages provided for splitting")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(pages)
    print(f"Split into {len(documents)} chunks.")
    return documents

# Generate embeddings with Ollama
def create_embeddings(documents):
    if not documents:
        raise ValueError("No documents provided for embedding generation")
    
    try:
        embedding_model = OllamaEmbeddings(
            model="gemma:2b",
            base_url="http://localhost:11434"
        )
        vector_store = FAISS.from_documents(documents, embedding_model)
        return vector_store
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        raise

# Set up Conversational Retrieval Chain
def setup_chatbot(vector_store):
    if not vector_store:
        raise ValueError("Vector store is required for chatbot setup")
    
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="mixtral-8x7b-32768",  # Updated model name
        temperature=0.7
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    return qa_chain

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global documents, vector_store, chatbot
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        file_path = os.path.join(os.getcwd(), "uploaded.pdf")
        file.save(file_path)
        print(f"File saved to {file_path}")
        
        pages = load_pdf(file_path)
        documents = split_text(pages)
        vector_store = create_embeddings(documents)
        chatbot = setup_chatbot(vector_store)
        
        return jsonify({
            "message": "PDF uploaded and processed successfully!",
            "pages": len(pages),
            "chunks": len(documents)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global chatbot
    
    if chatbot is None:
        return jsonify({"error": "Please upload a PDF first"}), 400
    
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
        
        user_input = data['message']
        try:
            response = chatbot.invoke({
                "question": user_input
            })
            
            return jsonify({
                "response": response.get('answer', ''),
                "source_documents": [doc.page_content for doc in response.get('source_documents', [])]
            })
            
        except Exception as chat_error:
            print(f"Chat error: {str(chat_error)}")
            return jsonify({"error": "Error processing chat request"}), 500
    
    except Exception as e:
        print(f"Request error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)