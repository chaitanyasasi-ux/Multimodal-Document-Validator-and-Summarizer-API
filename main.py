from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, Union, List
import io

# Import services
from services.ocr_service import image_to_text
from services.llm_service import LLMService

app = FastAPI(
    title="Multimodal Document Agent API",
    description="Processes text or image documents with OCR, guardrails, and summarization.",
    version="1.0"
)

# Initialize services
llm_service = LLMService()

# --- Pydantic Schemas for API Response ---
class ProcessingResult(BaseModel):
    is_valid: bool
    status_message: str
    extracted_text: str
    summary_points: Optional[List[str]] = None

class ErrorResponse(BaseModel):
    detail: str
    error_type: str

@app.post(
    "/process-document",
    response_model=ProcessingResult,
    responses={400: {"model": ErrorResponse}}
)
async def process_document(
    # Use Form for text input, File for image input
    text_input: Optional[str] = Form(None, description="Direct text input."),
    file: Optional[UploadFile] = File(None, description="Image file (.png, .jpg, etc.) for OCR.")
):
    """
    Accepts either a text string or an image file.
    1. Extracts text (via OCR if an image).
    2. Runs an AI Guardrail check (validation).
    3. If valid, runs an AI Summarization.
    """
    
    document_text = ""
    
    # 1. Handle Multimodal Input (Text or Image/OCR)
    if file:
        try:
            # Read the file content asynchronously
            image_bytes = await file.read()
            # Pass bytes to OCR service
            document_text = image_to_text(image_bytes)
            if document_text.startswith("OCR processing failed"):
                 # OCR failed, raise an error
                raise HTTPException(status_code=400, detail=ErrorResponse(
                    detail=document_text, 
                    error_type="OCR_FAILED"
                ).model_dump())
            
        except Exception as e:
            # Catch file read errors
            raise HTTPException(status_code=400, detail=ErrorResponse(
                detail=f"Error reading uploaded file: {str(e)}", 
                error_type="FILE_READ_ERROR"
            ).model_dump())
            
    elif text_input:
        document_text = text_input
        
    else:
        # No input provided
        raise HTTPException(status_code=400, detail=ErrorResponse(
            detail="Must provide either 'text_input' (Form) or 'file' (UploadFile).", 
            error_type="MISSING_INPUT"
        ).model_dump())
    
    if not document_text or document_text.upper() == "NO READABLE TEXT FOUND.":
        raise HTTPException(status_code=400, detail=ErrorResponse(
            detail="The provided document/image contains no readable text.", 
            error_type="EMPTY_CONTENT"
        ).model_dump())

    # 2. Run AI Guardrail Check (Chained LLM 1)
    is_valid, validation_reason = llm_service.validate_content(document_text)

    if not is_valid:
        # Guardrail Failed: Block processing and return error
        status_msg = f"Guardrail failed. Content blocked because: {validation_reason}."
        return ProcessingResult(
            is_valid=False,
            status_message=status_msg,
            extracted_text=document_text,
            summary_points=None
        )

    # 3. Run AI Summarization (Chained LLM 2 - only if Chain 1 passed)
    try:
        summary_raw = llm_service.summarize_content(document_text)
        summary_points = [p.strip("* ").strip("- ") for p in summary_raw.split('\n') if p.strip()]
        
        return ProcessingResult(
            is_valid=True,
            status_message="Processing successful. Content validated and summarized.",
            extracted_text=document_text,
            summary_points=summary_points
        )
    except Exception as e:
        # Catch LLM execution failure after validation
        raise HTTPException(status_code=500, detail=ErrorResponse(
            detail=f"LLM Summarization failed: {str(e)}", 
            error_type="LLM_EXECUTION_ERROR"
        ).model_dump())


# --- Run the application (Local Dev) ---
# if __name__ == "__main__":
#     import uvicorn
#     # You will run this and then use ngrok to expose it
#     uvicorn.run(app, host="0.0.0.0", port=8000)