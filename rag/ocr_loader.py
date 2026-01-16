from typing import List, Dict, Any
from .meta import process_document

def extract_text_with_ocr(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Converts OCR + tables into RAG-friendly text
    """
    pages = process_document(pdf_path)

    output = []
    for page in pages:
        text_parts = []

        # Normal text
        for block in page.text_blocks:
            text_parts.append(block)

        # Tables → convert rows to readable text
        for table in page.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text for cell in row)
                text_parts.append(f"TABLE_ROW: {row_text}")

        output.append({
            "page": page.page_number - 1,
            "text": "\n".join(text_parts),
            "sections": []
        })

    return output
