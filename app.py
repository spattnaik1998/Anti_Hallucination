# app.py - Flask Backend for RAG Comparison
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, TypedDict
import json
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress telemetry and warnings at startup
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*API key must be provided.*")

# Core dependencies
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.utils.function_calling import convert_to_openai_tool

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global storage for retrievers (in production, use Redis or database)
retrievers = {}

# =============================================================================
# RAG CLASSES (Same as your implementation)
# =============================================================================

# Data models for Self-RAG
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

# Self-RAG State
class SelfRAGState(TypedDict):
    """Represents the state of Self-RAG graph."""
    question: str
    generation: str
    documents: List[str]

class SelfRAGSystem:
    def __init__(self, retriever):
        self.retriever = retriever
        self.steps = []
        self.setup_chains()
    
    def add_step(self, title, content):
        """Add a step to track workflow"""
        self.steps.append({"title": title, "content": content})
    
    def setup_chains(self):
        """Setup all the chains for Self-RAG with optimized models"""
        # Use faster LLM for quick processing
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Faster, cheaper model
        
        # Document grader
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])
        self.retrieval_grader = grade_prompt | structured_llm_grader
        
        # RAG chain with faster model
        prompt = hub.pull("rlm/rag-prompt")
        rag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Use mini for speed
        self.rag_chain = prompt | rag_llm | StrOutputParser()
        
        # Hallucination grader
        hallucination_llm = llm.with_structured_output(GradeHallucinations)
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
             Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        
        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ])
        self.hallucination_grader = hallucination_prompt | hallucination_llm
        
        # Answer grader
        answer_llm = llm.with_structured_output(GradeAnswer)
        system = """You are a grader assessing whether an answer addresses / resolves a question \n
             Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ])
        self.answer_grader = answer_prompt | answer_llm
        
        # Question rewriter
        system = """You a question re-writer that converts an input question to a better version that is optimized \n
             for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ])
        self.question_rewriter = re_write_prompt | llm | StrOutputParser()
    
    def retrieve(self, state):
        """Retrieve documents"""
        self.add_step("🔍 Retrieve", "Retrieving relevant documents from vector store")
        question = state["question"]
        documents = self.retriever.invoke(question)
        self.add_step("📄 Retrieved", f"Found {len(documents)} documents")
        return {"documents": documents, "question": question}

    def generate(self, state):
        """Generate answer"""
        self.add_step("📝 Generate", "Generating answer from retrieved documents")
        question = state["question"]
        documents = state["documents"]
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(self, state):
        """Determines whether the retrieved documents are relevant to the question."""
        self.add_step("⚖️ Grade Documents", "Assessing document relevance to question")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke({
                "question": question, 
                "document": d.page_content
            })
            grade = score.binary_score
            if grade == "yes":
                filtered_docs.append(d)
        
        self.add_step("✅ Relevance Check", f"{len(filtered_docs)}/{len(documents)} documents deemed relevant")
        return {"documents": filtered_docs, "question": question}

    def transform_query(self, state):
        """Transform the query to produce a better question."""
        self.add_step("🔄 Transform Query", "Rewriting question for better retrieval")
        question = state["question"]
        documents = state["documents"]
        better_question = self.question_rewriter.invoke({"question": question})
        self.add_step("✏️ New Query", f"Transformed to: {better_question}")
        return {"documents": documents, "question": better_question}

    def decide_to_generate(self, state):
        """Determines whether to generate an answer, or re-generate a question."""
        filtered_documents = state["documents"]

        if not filtered_documents:
            self.add_step("❌ No Relevant Docs", "All documents filtered out, transforming query")
            return "transform_query"
        else:
            self.add_step("✅ Proceed", "Relevant documents found, proceeding to generate")
            return "generate"

    def grade_generation_v_documents_and_question(self, state):
        """Determines whether the generation is grounded in the document and answers question."""
        self.add_step("🤔 Self-Reflection", "Checking for hallucinations and answer quality")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke({
            "documents": documents, 
            "generation": generation
        })
        grade = score.binary_score

        if grade == "yes":
            self.add_step("✅ Grounded", "Answer is grounded in documents")
            score = self.answer_grader.invoke({
                "question": question, 
                "generation": generation
            })
            grade = score.binary_score
            if grade == "yes":
                self.add_step("🎯 Quality Check", "Answer addresses the question well")
                return "useful"
            else:
                self.add_step("❌ Poor Quality", "Answer doesn't address question, refining")
                return "not useful"
        else:
            self.add_step("⚠️ Hallucination", "Answer not grounded in documents, regenerating")
            return "not supported"

    def process_question(self, question):
        """Process question through Self-RAG workflow"""
        self.steps = []
        final_answer = "No answer generated"
        
        try:
            workflow = StateGraph(SelfRAGState)
            workflow.add_node("retrieve", self.retrieve)
            workflow.add_node("grade_documents", self.grade_documents)
            workflow.add_node("generate", self.generate)
            workflow.add_node("transform_query", self.transform_query)

            workflow.add_edge(START, "retrieve")
            workflow.add_edge("retrieve", "grade_documents")
            workflow.add_conditional_edges(
                "grade_documents",
                self.decide_to_generate,
                {
                    "transform_query": "transform_query",
                    "generate": "generate",
                },
            )
            workflow.add_edge("transform_query", "retrieve")
            workflow.add_conditional_edges(
                "generate",
                self.grade_generation_v_documents_and_question,
                {
                    "not supported": "generate",
                    "useful": END,
                    "not useful": "transform_query",
                },
            )

            app_workflow = workflow.compile()
            
            inputs = {"question": question}
            final_result = None
            
            for output in app_workflow.stream(inputs):
                final_result = output
                
            # Extract final generation
            if final_result:
                for key, value in final_result.items():
                    if isinstance(value, dict) and "generation" in value:
                        final_answer = value["generation"]
                        self.add_step("🎯 Final Answer", final_answer)
                        break
            
            if final_answer == "No answer generated":
                final_answer = "I apologize, but I couldn't generate a satisfactory answer based on the provided documents. This might be because the documents don't contain relevant information for your question, or the query processing encountered an issue."
                self.add_step("⚠️ No Answer Generated", final_answer)
                
        except Exception as e:
            error_msg = f"Error in Self-RAG processing: {str(e)}"
            self.add_step("❌ Error", error_msg)
            final_answer = "An error occurred while processing your question with Self-RAG. Please try rephrasing your question or check if the uploaded documents contain relevant information."
        
        return {
            "steps": self.steps,
            "final_answer": final_answer
        }

# Corrective RAG State
class CorrectiveRAGState(TypedDict):
    """Represents the state of Corrective RAG graph."""
    keys: Dict[str, any]

class CorrectiveRAGSystem:
    def __init__(self, retriever):
        self.retriever = retriever
        self.steps = []
        self.setup_chains()
    
    def add_step(self, title, content):
        """Add a step to track workflow"""
        self.steps.append({"title": title, "content": content})
    
    def setup_chains(self):
        """Setup chains for Corrective RAG with optimized models"""
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)  # Faster model
        
        # RAG prompt
        self.rag_prompt = hub.pull("rlm/rag-prompt")
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()
        
        # Grading model - use faster model
        class Grade(BaseModel):
            """Binary score for relevance check."""
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")
        
        model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)  # Faster model
        grade_tool_oai = convert_to_openai_tool(Grade)
        
        self.llm_with_tool = model.bind(
            tools=[convert_to_openai_tool(grade_tool_oai)],
            tool_choice={"type": "function", "function": {"name": "Grade"}},
        )
        
        self.parser_tool = PydanticToolsParser(tools=[Grade])
        
        # Grading prompt
        self.grade_prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )
        
        self.grading_chain = self.grade_prompt | self.llm_with_tool | self.parser_tool
        
        # Query transformation
        self.transform_prompt = PromptTemplate(
            template="""You are generating questions that is well optimized for retrieval. \n
            Look at the input and try to reason about the underlying sematic intent / meaning. \n
            Here is the initial question:
            \n --------- \n
            {question}
            \n --------- \n
            Formulate an improved question: """,
            input_variables=["question"],
        )
        
        self.transform_chain = self.transform_prompt | self.llm | StrOutputParser()

    def retrieve(self, state):
        """Helper function for retrieving documents"""
        self.add_step("🔍 Retrieve", "Retrieving documents from vector store")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = self.retriever.get_relevant_documents(question)
        self.add_step("📄 Retrieved", f"Found {len(documents)} documents")
        return {"keys": {"documents": documents, "question": question}}

    def generate(self, state):
        """Helper function for generating answers"""
        self.add_step("📝 Generate", "Generating final answer")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {
            "keys": {"documents": documents, "question": question, "generation": generation}
        }

    def grade_documents(self, state):
        """Determines whether the retrieved documents are relevant to the question."""
        self.add_step("⚖️ Grade Documents", "Assessing document relevance")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        filtered_docs = []
        search = "No"  # Default do not opt for web search
        
        for d in documents:
            score = self.grading_chain.invoke({
                "question": question, 
                "context": d.page_content
            })
            grade = score[0].binary_score
            if grade == "yes":
                filtered_docs.append(d)
            else:
                search = "Yes"  # Perform web search when documents are not relevant

        self.add_step("✅ Relevance Check", f"{len(filtered_docs)}/{len(documents)} documents relevant. Web search: {search}")
        
        return {
            "keys": {
                "documents": filtered_docs,
                "question": question,
                "run_web_search": search,
            }
        }

    def transform_query(self, state):
        """Helper function for transforming the query to produce a better question."""
        self.add_step("🔄 Transform Query", "Optimizing query for web search")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        better_question = self.transform_chain.invoke({"question": question})
        self.add_step("✏️ New Query", f"Transformed to: {better_question}")
        return {"keys": {"documents": documents, "question": better_question}}

    def web_search(self, state):
        """Helper function to do Web search based on the re-phrased question using Tavily API."""
        try:
            self.add_step("🌐 Web Search", "Searching web for additional context")
            state_dict = state["keys"]
            question = state_dict["question"]
            documents = state_dict["documents"]

            tool = TavilySearchResults()
            docs = tool.invoke({"query": question})
            web_results = "\n".join([d["content"] for d in docs])
            web_results = Document(page_content=web_results)
            documents.append(web_results)

            self.add_step("📡 Web Results", f"Added web search results to context")
            return {"keys": {"documents": documents, "question": question}}
        except Exception as e:
            self.add_step("⚠️ Web Search Failed", f"Web search encountered an error: {str(e)}")
            # Continue without web results
            state_dict = state["keys"]
            return {"keys": {"documents": state_dict["documents"], "question": state_dict["question"]}}

    def decide_to_generate(self, state):
        """Helper function to determine whether to generate an answer or re-generate a question for web search."""
        state_dict = state["keys"]
        search = state_dict["run_web_search"]

        if search == "Yes":
            self.add_step("🚀 Web Search Path", "Insufficient relevant docs, proceeding to web search")
            return "transform_query"
        else:
            self.add_step("✅ Direct Generation", "Sufficient relevant docs, generating answer")
            return "generate"

    def process_question(self, question):
        """Process question through Corrective RAG workflow"""
        self.steps = []
        final_answer = "No answer generated"
        
        try:
            workflow = StateGraph(CorrectiveRAGState)
            workflow.add_node("retrieve", self.retrieve)
            workflow.add_node("grade_documents", self.grade_documents)
            workflow.add_node("generate", self.generate)
            workflow.add_node("transform_query", self.transform_query)
            workflow.add_node("web_search", self.web_search)

            workflow.set_entry_point("retrieve")
            workflow.add_edge("retrieve", "grade_documents")
            workflow.add_conditional_edges(
                "grade_documents",
                self.decide_to_generate,
                {
                    "transform_query": "transform_query",
                    "generate": "generate",
                },
            )
            workflow.add_edge("transform_query", "web_search")
            workflow.add_edge("web_search", "generate")
            workflow.add_edge("generate", END)

            app_workflow = workflow.compile()
            
            inputs = {"keys": {"question": question}}
            final_result = None
            
            for output in app_workflow.stream(inputs):
                final_result = output
                
            # Extract final generation
            if final_result:
                for key, value in final_result.items():
                    if isinstance(value, dict) and "keys" in value and isinstance(value["keys"], dict) and "generation" in value["keys"]:
                        final_answer = value["keys"]["generation"]
                        self.add_step("🎯 Final Answer", final_answer)
                        break
            
            if final_answer == "No answer generated":
                final_answer = "I apologize, but I couldn't generate a satisfactory answer based on the provided documents and web search. This might be because the available information doesn't contain relevant content for your question."
                self.add_step("⚠️ No Answer Generated", final_answer)
                
        except Exception as e:
            error_msg = f"Error in Corrective RAG processing: {str(e)}"
            self.add_step("❌ Error", error_msg)
            final_answer = "An error occurred while processing your question with Corrective RAG. Please try rephrasing your question or check if the uploaded documents contain relevant information."
        
        return {
            "steps": self.steps,
            "final_answer": final_answer
        }

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_pdfs_from_folder(folder_path: str):
    """Load all PDF documents from the specified folder"""
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))
    
    docs = []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            pdf_docs = loader.load()
            docs.extend(pdf_docs)
        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")
    
    return docs

def create_vectorstore_from_pdfs(folder_path: str):
    """Create vector store from PDF documents in folder with optimized settings"""
    docs_list = load_pdfs_from_folder(folder_path)
    
    if not docs_list:
        raise ValueError("No documents were successfully loaded from PDFs")
    
    print(f"Splitting {len(docs_list)} documents into chunks...")
    
    # Optimized chunking for faster processing
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,  # Larger chunks = fewer embeddings to compute
        chunk_overlap=50  # Small overlap for speed
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    # Limit total chunks for faster processing
    max_chunks = 1000  # Reasonable limit for fast processing
    if len(doc_splits) > max_chunks:
        print(f"Limiting to {max_chunks} chunks (was {len(doc_splits)} chunks) for faster processing")
        doc_splits = doc_splits[:max_chunks]
    
    if not doc_splits:
        raise ValueError("No text chunks were created from the documents")
    
    print(f"Creating vector store with {len(doc_splits)} chunks...")
    
    # Suppress ChromaDB telemetry by setting environment variable
    os.environ["ANONYMIZED_TELEMETRY"] = "false"
    
    # Use optimized ChromaDB settings
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=f"pdf-rag-{uuid.uuid4().hex[:8]}",
        embedding=OpenAIEmbeddings(
            # Use smaller, faster embedding model
            model="text-embedding-3-small"
        ),
        persist_directory=None  # In-memory for faster processing
    )
    
    print("Vector store created successfully!")
    return vectorstore.as_retriever(search_kwargs={"k": 4})  # Retrieve fewer docs for speed

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle PDF file uploads with progress tracking"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400
    
    # Create session-specific upload folder
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(upload_path, exist_ok=True)
    
    uploaded_files = []
    total_size = 0
    max_file_size = 10 * 1024 * 1024  # 10MB per file
    max_total_files = 10  # Limit number of files
    
    # Filter and validate files
    pdf_files = [f for f in files if f and f.filename.endswith('.pdf')]
    
    if len(pdf_files) > max_total_files:
        return jsonify({'error': f'Too many files. Maximum {max_total_files} PDFs allowed for optimal performance.'}), 400
    
    for file in pdf_files:
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > max_file_size:
            return jsonify({'error': f'File {file.filename} is too large. Maximum 10MB per file.'}), 400
        
        total_size += file_size
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_path, filename)
        file.save(filepath)
        uploaded_files.append(filename)
    
    if not uploaded_files:
        return jsonify({'error': 'No valid PDF files uploaded'}), 400
    
    try:
        print(f"Processing {len(uploaded_files)} PDF files...")
        # Create retriever and store it
        retriever = create_vectorstore_from_pdfs(upload_path)
        retrievers[session_id] = retriever
        
        return jsonify({
            'success': True,
            'files': uploaded_files,
            'session_id': session_id,
            'message': f'Successfully processed {len(uploaded_files)} PDF files'
        })
    except Exception as e:
        # Clean up on error
        if os.path.exists(upload_path):
            shutil.rmtree(upload_path)
        return jsonify({'error': f'Error processing PDFs: {str(e)}'}), 500

@app.route('/query', methods=['POST'])
def process_query():
    """Process query with both RAG systems"""
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    session_id = session.get('session_id')
    if not session_id or session_id not in retrievers:
        return jsonify({'error': 'No PDF documents uploaded. Please upload PDFs first.'}), 400
    
    try:
        retriever = retrievers[session_id]
        
        # Initialize both systems
        self_rag = SelfRAGSystem(retriever)
        corrective_rag = CorrectiveRAGSystem(retriever)
        
        # Process with both systems
        self_rag_result = self_rag.process_question(question)
        corrective_rag_result = corrective_rag.process_question(question)
        
        return jsonify({
            'success': True,
            'self_rag': self_rag_result,
            'corrective_rag': corrective_rag_result
        })
        
    except Exception as e:
        error_msg = f'Error processing query: {str(e)}'
        print(f"Query processing error: {e}")  # Log for debugging
        
        # Return error response with fallback results
        return jsonify({
            'success': False,
            'error': error_msg,
            'self_rag': {
                'steps': [{'title': '❌ Error', 'content': error_msg}],
                'final_answer': 'An error occurred while processing your question. Please try again with a different question.'
            },
            'corrective_rag': {
                'steps': [{'title': '❌ Error', 'content': error_msg}],
                'final_answer': 'An error occurred while processing your question. Please try again with a different question.'
            }
        }), 500

@app.route('/clear', methods=['POST'])
def clear_session():
    """Clear uploaded files and session"""
    session_id = session.get('session_id')
    if session_id:
        # Remove uploaded files
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if os.path.exists(upload_path):
            shutil.rmtree(upload_path)
        
        # Remove from retrievers
        if session_id in retrievers:
            del retrievers[session_id]
        
        session.clear()
    
    return jsonify({'success': True})

if __name__ == '__main__':
    # Check for required environment variables
    openai_key = os.getenv('OPENAI_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    
    if not openai_key:
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        print("Please create a .env file with your API keys:")
        print("OPENAI_API_KEY=your-openai-key-here")
        print("TAVILY_API_KEY=your-tavily-key-here")
        exit(1)
    
    if not tavily_key:
        print("❌ Error: TAVILY_API_KEY not found in .env file")
        print("Please add TAVILY_API_KEY to your .env file:")
        print("TAVILY_API_KEY=your-tavily-key-here")
        exit(1)
    
    print("✅ API keys loaded successfully from .env file")
    print(f"🚀 Starting Flask application...")
    print(f"📖 Open your browser and go to: http://localhost:5000")
    
    app.run(debug=True, port=5000)