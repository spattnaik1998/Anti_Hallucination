# Anti_Hallucination

# 🤖 Self-RAG vs Corrective RAG Comparison Tool

A comprehensive web application that demonstrates and compares two advanced agentic RAG (Retrieval-Augmented Generation) workflows side-by-side using your PDF documents. Built with Flask backend and modern HTML/CSS/JS frontend.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-purple.svg)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com)

## 🌟 Features

### 🔄 Self-RAG Workflow
- **Iterative refinement** with self-reflection loops
- **Document relevance grading** at each step
- **Hallucination detection** and correction
- **Query transformation** when documents are insufficient
- **Multi-iteration processing** until satisfactory answer

### 🔧 Corrective RAG Workflow
- **Single-pass processing** with intelligent routing
- **Document relevance assessment** with filtering
- **Conditional web search** when PDFs lack information
- **Query optimization** for web search
- **Hybrid context** combining PDFs + web results

### 🎯 Core Capabilities
- **PDF Document Processing** with text extraction and chunking
- **Vector Search** using ChromaDB and OpenAI embeddings
- **Real-time Visualization** of workflow steps
- **Side-by-side Comparison** of both approaches
- **Session Management** with file isolation
- **Performance Optimization** for fast processing

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Tavily API key ([Get one here](https://tavily.com))

### Installation

1. **Clone or download the project**
   ```bash
   git clone <your-repo-url>
   cd rag-comparison-app
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your API keys:
   # OPENAI_API_KEY=sk-your-actual-openai-key
   # TAVILY_API_KEY=tvly-your-actual-tavily-key
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
rag-comparison-app/
├── 📄 app.py                    # Flask backend with RAG implementations
├── 📄 requirements.txt          # Python dependencies
├── 📄 .env                     # Your API keys (create from .env.example)
├── 📄 .env.example            # Template for environment variables
├── 📄 .gitignore             # Git ignore patterns
├── 📄 README.md             # This documentation
│
├── 📁 templates/            # Flask templates
│   └── 📄 index.html       # Frontend interface (HTML/CSS/JS)
│
└── 📁 uploads/             # Temporary PDF storage (auto-created)
    ├── 📁 session-abc123/  # User session folders
    └── 📁 session-def456/  # Isolated file storage
```

## 🔑 API Keys Setup

### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Add to your `.env` file: `OPENAI_API_KEY=sk-your-key-here`

### Tavily API Key
1. Visit [Tavily](https://tavily.com)
2. Sign up for an account
3. Get your API key
4. Add to your `.env` file: `TAVILY_API_KEY=tvly-your-key-here`

### Example .env file
```bash
# OpenAI API Key - Required for LLM operations
OPENAI_API_KEY=sk-proj-your-actual-openai-key-here

# Tavily API Key - Required for web search in Corrective RAG
TAVILY_API_KEY=tvly-your-actual-tavily-key-here
```

## 📖 How to Use

### 1. Upload PDF Documents
- **Drag & Drop**: Drag PDF files directly onto the upload area
- **Click to Select**: Click "Choose PDF Files" to browse and select
- **Multiple Files**: Upload up to 10 PDF files at once
- **File Validation**: Automatic validation for PDF format and file size

### 2. Wait for Processing
- **Text Extraction**: PDF content is extracted and processed
- **Chunking**: Documents are split into manageable segments
- **Vector Embeddings**: Text chunks are converted to embeddings
- **Progress Tracking**: Real-time progress updates in console

### 3. Ask Questions
- **Natural Language**: Ask questions in plain English
- **Document-Specific**: Questions about the uploaded PDF content
- **Comparative**: Ask for comparisons or summaries across documents
- **Press Enter**: Quick submission with Enter key

### 4. Compare Results
- **Self-RAG Panel**: Shows iterative refinement process
- **Corrective RAG Panel**: Shows single-pass with web augmentation
- **Step-by-Step**: Real-time visualization of each workflow step
- **Final Answers**: Compare the quality and approach of both systems

## 🎯 Example Questions

### Research Papers
- "What is the main research question addressed in these papers?"
- "Compare the methodologies used across the different studies"
- "What are the key findings and their statistical significance?"
- "What limitations do the authors acknowledge?"

### Business Documents
- "Summarize the quarterly financial performance"
- "What are the main strategic initiatives mentioned?"
- "Compare the revenue projections across different quarters"
- "What risks and challenges are identified?"

### Technical Documentation
- "Explain the system architecture described in the documents"
- "What are the technical requirements and dependencies?"
- "Compare the different implementation approaches"
- "What troubleshooting steps are recommended?"

## 🔄 Workflow Comparison

### Self-RAG Process
```
1. Initial Query
   ↓
2. Document Retrieval
   ↓
3. Document Relevance Grading
   ↓
4. Generate Answer (if docs relevant)
   ↓
5. Self-Reflection
   ├── Hallucination Check
   └── Answer Quality Assessment
   ↓
6. Decision Point:
   ├── Accept Answer (if good)
   ├── Regenerate (if hallucinated)
   └── Transform Query (if poor quality)
```

### Corrective RAG Process
```
1. Initial Query
   ↓
2. Document Retrieval
   ↓
3. Document Relevance Assessment
   ↓
4. Decision Point:
   ├── Generate Answer (if sufficient docs)
   └── Web Search Path:
       ├── Transform Query
       ├── Web Search
       └── Generate with Combined Context
```

## ⚡ Performance Optimizations

### File Limits (for speed)
- **Maximum Files**: 10 PDF files per upload
- **File Size**: 10MB maximum per PDF
- **Page Limit**: First 20 pages per PDF (for large documents)
- **Total Chunks**: Maximum 1000 text chunks across all documents

### Model Optimizations
- **LLM Model**: GPT-4o-mini (faster than GPT-4o)
- **Embedding Model**: text-embedding-3-small (faster, cheaper)
- **Chunk Size**: 500 tokens (larger chunks = fewer embeddings)
- **Retrieval**: Top 4 documents (reduced for speed)

### Processing Speed
- **Small PDFs (1-5 pages)**: ~10-30 seconds
- **Medium PDFs (10-20 pages)**: ~30-60 seconds
- **Multiple PDFs (5-10 files)**: ~1-3 minutes
- **Large documents**: Limited to first 20 pages automatically

## 🛠️ Technical Architecture

### Backend (Flask)
- **Framework**: Flask 2.3+ with session management
- **RAG Implementation**: LangChain + LangGraph for workflow orchestration
- **Vector Store**: ChromaDB with OpenAI embeddings
- **PDF Processing**: PyPDF for text extraction
- **Web Search**: Tavily API for external information

### Frontend (HTML/CSS/JS)
- **Interface**: Single-page application with real-time updates
- **File Upload**: Drag & drop with progress tracking
- **Visualization**: Step-by-step workflow display
- **Responsive**: Works on desktop and mobile devices

### Key Dependencies
```python
Flask==2.3.3              # Web framework
langchain==0.1.0          # LLM framework
langchain-openai==0.0.2   # OpenAI integration
langgraph==0.0.20         # Workflow orchestration
chromadb==0.4.18          # Vector database
pypdf==3.17.4             # PDF processing
tavily-python==0.3.0      # Web search
python-dotenv==1.0.0      # Environment variables
```

## 🚨 Troubleshooting

### Common Issues

#### 1. API Key Errors
```
Error: OPENAI_API_KEY not found in .env file
```
**Solution**: Create `.env` file with your API keys from `.env.example`

#### 2. PDF Processing Slow
```
PDFs taking too long to process
```
**Solution**: 
- Use smaller PDFs (under 10MB)
- Limit to 5-10 files per upload
- App automatically limits to first 20 pages

#### 3. ChromaDB Telemetry Errors
```
Failed to send telemetry event
```
**Solution**: Already handled in the code with environment variables

#### 4. Module Import Errors
```
ModuleNotFoundError: No module named 'langchain'
```
**Solution**: 
- Activate virtual environment: `source venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`

#### 5. Memory Issues
```
Out of memory during processing
```
**Solution**:
- Reduce number of PDFs
- Use smaller files
- Restart the application

### Performance Tips

#### For Faster Processing
- **Fewer files**: Upload 3-5 PDFs instead of 10
- **Smaller files**: Use PDFs under 5MB when possible
- **Text-based PDFs**: Avoid scanned/image-based PDFs
- **Relevant content**: Upload only documents related to your questions

#### For Better Results
- **Clear questions**: Ask specific, well-formed questions
- **Context-aware**: Reference the type of documents you uploaded
- **Iterative**: Start with broad questions, then get specific
- **Document quality**: Use well-structured, text-rich PDFs

## 🔐 Security & Privacy

### Data Handling
- **Temporary Storage**: PDF files stored temporarily during session
- **Automatic Cleanup**: Files deleted when session ends or manually cleared
- **Session Isolation**: Each user gets isolated file storage
- **No Persistence**: Vector stores created in-memory, not saved

### API Key Security
- **Environment Variables**: API keys stored in `.env` file
- **Git Ignored**: `.env` file automatically ignored in version control
- **Local Only**: Keys never transmitted or logged
- **User Responsibility**: Users manage their own API keys

### Privacy
- **Local Processing**: All PDF processing happens on your machine
- **API Calls**: Only text chunks sent to OpenAI for embeddings/generation
- **No Data Retention**: No user data stored permanently
- **Session-Based**: Each session is independent and temporary

## 🤝 Contributing

### Development Setup
1. **Fork** the repository
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes** and test thoroughly
4. **Commit changes**: `git commit -m 'Add amazing feature'`
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Open Pull Request** with detailed description

### Code Style
- **Python**: Follow PEP 8 guidelines
- **JavaScript**: Use ES6+ features
- **HTML/CSS**: Semantic markup and responsive design
- **Comments**: Document complex logic and API interactions

### Testing
- **Manual Testing**: Test with various PDF types and sizes
- **Error Handling**: Verify graceful failure modes
- **Cross-Browser**: Test in Chrome, Firefox, Safari
- **Mobile**: Verify responsive design works

## 📊 Comparison Results

### When to Use Self-RAG
- **High Accuracy Requirements**: When you need the most accurate possible answers
- **Complex Questions**: Multi-step reasoning or analysis
- **Quality over Speed**: When answer quality is more important than response time
- **Iterative Refinement**: When the system should improve its own responses

### When to Use Corrective RAG
- **Speed Requirements**: When you need faster responses
- **Broad Information Needs**: When documents might not contain all necessary info
- **Web Augmentation**: When external information can enhance answers
- **Single-Pass Processing**: When you prefer straightforward, direct responses

### Expected Differences
- **Self-RAG**: More steps, higher quality, slower processing
- **Corrective RAG**: Fewer steps, web-enhanced, faster processing
- **Use Cases**: Self-RAG for analysis, Corrective RAG for information gathering

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the excellent RAG framework
- **OpenAI**: For powerful language models and embeddings
- **ChromaDB**: For efficient vector storage
- **Tavily**: For web search capabilities
- **Flask**: For the lightweight web framework

## 📞 Support

### Issues
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check this README for common solutions
- **Community**: Share experiences and solutions

### Resources
- **LangChain Docs**: [langchain.com](https://langchain.com)
- **OpenAI API**: [platform.openai.com](https://platform.openai.com)
- **Flask Documentation**: [flask.palletsprojects.com](https://flask.palletsprojects.com)

---

## 🚀 Get Started Now!

1. **Clone the repository**
2. **Set up your environment** (Python 3.8+)
3. **Install dependencies** (`pip install -r requirements.txt`)
4. **Configure API keys** (create `.env` from `.env.example`)
5. **Run the application** (`python app.py`)
6. **Upload your PDFs** and start comparing RAG workflows!

---

*Built with ❤️ for the RAG community. Happy experimenting!* 🤖
