# Multimodal-Document-Validator-and-Summarizer-API
Multimodal AI Document Agent API (Validator & Summarizer)
🌟 Project Overview
This project implements a resilient, two-stage Large Language Model (LLM) pipeline exposed as a high-performance API using FastAPI. It's designed to process documents from both text and image sources, ensuring content safety and professionalism before generating a concise summary.

The core value lies in demonstrating MLOps-centric engineering concepts crucial for production environments.

✨ Core Features & Advanced Engineering
Category,Feature,Technical Skill Demonstrated
Pipeline Architecture,LLM Chaining (Guardrail),The system executes a conditional workflow: Validate → Summarize. It uses the first LLM call as a safety check before triggering the more expensive summarization.
Multimodal Input,Integrated OCR,"Accepts raw text or image files (.png, .jpg, etc.) and uses EasyOCR for text extraction."
Performance,Model Caching,"The resource-intensive EasyOCR model instance is loaded and cached in memory only once at startup, significantly reducing latency for subsequent image processing requests."
Reliability,Exponential Backoff & Retries,"Implements robust logic to catch transient API errors (e.g., 503 Service Unavailable) and automatically retry the call with increasing delay, maximizing uptime."
API Design,Pydantic Schemas & Structured Errors,"Enforces clear data contracts for inputs/outputs and returns specific, machine-readable error codes (OCR_FAILED, MISSING_INPUT, etc.)."

🛠️ Technology Stack
Component,Technology,Role
API Framework,FastAPI,High-performance API routing and serving.
LLM Backend,Google Gemini API,Core intelligence for validation and summarization.
Image Processing,"EasyOCR, Pillow",Optical Character Recognition (OCR) and image handling.
Environment,python-dotenv,Secure management of API keys and configuration.
Server,Uvicorn,ASGI server for asynchronous request handling.

🚀 Getting Started
Prerequisites
Python 3.8+

An active Google Gemini API Key.

Setup Instructions
Clone the repository:
git clone https://github.com/chaitanyasasi-ux/Multimodal-Document-Validator-and-Summarizer-API.git
cd Multimodal-Document-Validator-and-Summarizer-APIhe repository
Create and activate a virtual environment:
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
Install dependencies: (Ensure you have a requirements.txt file listing all libraries: fastapi, pydantic, easyocr, google-genai, python-dotenv, uvicorn[standard])
pip install -r requirements.txt
Configure API Key:

Create a file named .env in the root directory.

Add your Gemini API key to the file:
GEMINI_API_KEY="YOUR_API_KEY_HERE"
Running the API
Start the Uvicorn server:
uvicorn main:app --reload
The API is now running at: http://127.0.0.1:8000
💻 Usage
The interactive API documentation (Swagger UI) is available at: http://127.0.0.1:8000/docs

Endpoint: POST /process-document
Processes a document using either direct text or an uploaded image.
Input,Type,Description,Example Use Case
text_input,Form (string),Direct text content. Must be provided if file is missing.,Validating a pasted contract draft.
file,UploadFile (image),"Image file (JPG, PNG, etc.). Must be provided if text_input is missing.",Summarizing a scanned page or document.

Example Success Response (200 OK)
{
  "is_valid": true,
  "status_message": "Processing successful. Content validated and summarized.",
  "extracted_text": "The quick brown fox jumps over...",
  "summary_points": [
    "1. Document successfully passed the safety guardrail.",
    "2. Key topic analyzed was content validation.",
    "3. Summary generated concisely in three points."
  ]
}
💡 Future Enhancements
The project is structured for easy extension. Potential next steps include:

Dockerization: Adding a Dockerfile for easy deployment via tools like AWS ECS or GCP Cloud Run.

Structured Logging: Implementing JSON logging for production monitoring and analysis.

Database Integration: Connecting to MySQL (as per project skills) to persist results.

Asynchronous LLM Client: Refactoring LLM calls to use a true async client for higher concurrency.

📄 License
This project is licensed under the MIT License.

