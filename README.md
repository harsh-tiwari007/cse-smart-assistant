 CSE Smart Assistant (Offline AI Chatbot)

CSE Smart Assistant is an offline AI-based chatbot designed to answer queries related to the Computer Science Department. It uses semantic search and a local language model to generate accurate responses from provided data sources such as text files and PDFs.

---

##  Features

-  Supports both **TXT files and PDF documents**
-  Semantic search using **FAISS + Sentence Transformers**
-  AI response generation using **TinyLlama (offline model)**
-  Fully offline (no API required)
-  Multiple data sources:
  - Default data (TXT files)
  - Manual input data
  - Uploaded PDF
-  Chat history tracking
-  Option to clear chat history
-  Domain-specific answering (CSE focused)

---

##  Tech Stack

- Frontend: Streamlit  
- Backend: Python  
- AI Model: TinyLlama (HuggingFace Transformers)  
- Embeddings: Sentence Transformers  
- Vector Database: FAISS  
- PDF Processing: PyPDF  

---

##  Project Structure


cse-smart-assistant/
│
├── app.py
├── data/
│ ├── about.txt
│ ├── faculty.txt
│ ├── labs.txt
│ ├── eoa_report.txt
│ └── (PDF files)
│
├── .gitignore
└── README.md


---

##  How It Works

1. Loads data from TXT files and/or uploaded PDF  
2. Splits data into smaller chunks  
3. Converts text into embeddings using SentenceTransformer  
4. Stores embeddings in FAISS vector database  
5. User query is converted into embedding  
6. Most relevant data chunk is retrieved  
7. Context + query is passed to TinyLlama model  
8. Model generates a relevant answer  

---

##  How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/harsh-tiwari007/cse-smart-assistant.git
cd cse-smart-assistant
2. Create virtual environment
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install streamlit transformers sentence-transformers faiss-cpu pypdf
4. Run the application
streamlit run app.py
💡 Example Questions
Who is the HOD of CSE department?
What courses are offered in CSE?
What labs are available?
Show syllabus details
🎯 Use Cases
College department chatbot
Academic query assistant
Offline AI system for institutions
🔒 Limitations
Works only on provided data
No internet-based knowledge
Limited by model capability
🚀 Future Improvements
Better UI/UX
Fine-tuned custom model
Multi-department support
Voice assistant integration
👨‍💻 Author

Harsh Tiwari

---screenshots
<img width="1919" height="903" alt="Screenshot 2026-03-28 202712" src="https://github.com/user-attachments/assets/15d0d938-8738-498c-b704-2b06fec0afa8" />
<img width="1919" height="954" alt="Screenshot 2026-03-28 202115" src="https://github.com/user-attachments/assets/6ddaca78-626e-4a84-a775-837bb664814e" />
