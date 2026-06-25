# OCR Utilities For Conversational RAG + OCR Project
import os
import fitz
from paddleocr import PaddleOCR
from langchain_community.document_loaders import PyPDFLoader
# Initialize OCR Model Once
ocr=PaddleOCR(use_angle_cls=True,lang="en")
# Create Temp Image Directory
TEMP_IMAGE_DIR="Temp_Images"
os.makedirs(TEMP_IMAGE_DIR,exist_ok=True)
def extract_text_from_pdf(pdf_path):
    """
    Intelligent Extraction
    1. Try normal PDF extraction
    2. If no text found -> OCR
    """
    try:
        loader=PyPDFLoader(pdf_path)
        docs=loader.load()
        text="\n".join([doc.page_content for doc in docs])
        if len(text.strip())>100:
            print("Using Normal PDF Text Extraction")
            return text
    except Exception as e:
        print(f"PDF Extraction Failed: {e}")
    print("Using OCR Extraction")
    return extract_text_with_ocr(pdf_path)
def extract_text_with_ocr(pdf_path):
    """
    Extract text from a PDF using PaddleOCR.
    """
    extracted_text=""
    pdf=fitz.open(pdf_path)
    for page_num in range(len(pdf)):
        page=pdf.load_page(page_num)
        # Increase Resolution For Better OCR
        pix=page.get_pixmap(matrix=fitz.Matrix(2,2))
        image_path=os.path.join(TEMP_IMAGE_DIR,f"Page_{page_num}.png")
        pix.save(image_path)
        try:
            result=ocr.ocr(image_path)
            if result and result[0]:
                extracted_text+=f"\n\n--- Page {page_num + 1} ---\n\n"
                for line in result[0]:
                    text=line[1][0]
                    extracted_text+=text+"\n"
        except Exception as e:
            print(f"OCR Error On Page {page_num + 1}: {e}")
        finally:
            # Delete Temporary Image
            if os.path.exists(image_path):
                os.remove(image_path)
    pdf.close()
    return extracted_text
# Test OCR Independently
if __name__=="__main__":
    pdf_file="Temp-Scan.pdf"
    text=extract_text_from_pdf(pdf_file)
    print(text)