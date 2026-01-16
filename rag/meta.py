import os
import uuid
import json
import logging
import numpy as np
import fitz  # PyMuPDF
import cv2
import pytesseract
from PIL import Image
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from multiprocessing import Pool, cpu_count
from pathlib import Path
import re

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class OCRConfig:
    tesseract_cmd: str = ""
    dpi: int = 250  # Increased for better table detection
    parallel_workers: int = max(1, cpu_count() - 1)
    row_tolerance: int = 18
    col_gap_threshold: int = 35
    min_table_width_ratio: float = 0.20
    min_table_height: int = 45
    ocr_languages: str = "eng"
    psm_modes: List[int] = None  # Will default to multiple modes

    def __post_init__(self):
        if not self.psm_modes:
            self.psm_modes = [3, 6, 11]  # Better for tables and mixed content
        if not self.tesseract_cmd:
            self.tesseract_cmd = self._find_tesseract()

    def _find_tesseract(self) -> str:
        """Auto-detect Tesseract path across platforms."""
        possible_paths = [
            r"C:\Users\ISFL-RT000263\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "tesseract"  # Rely on PATH
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        logger.warning("Tesseract not found. Install Tesseract OCR.")
        return "tesseract"

@dataclass
class TableCell:
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: Optional[float] = None

@dataclass
class Table:
    bbox: Tuple[int, int, int, int]
    rows: List[List[TableCell]]
    nested_tables: List['Table'] = None
    confidence: float = 0.0

    def __post_init__(self):
        if self.nested_tables is None:
            self.nested_tables = []

@dataclass
class PageContent:
    page_number: int
    text_blocks: List[str]
    tables: List[Table]
    total_words: int
    ocr_confidence: float

class TableExtractor:
    def __init__(self, config: OCRConfig):
        self.config = config
    
    def deskew_image(self, img: np.ndarray) -> np.ndarray:
        """Deskew image to improve table detection."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, height2 = line[0]
                angle = np.arctan2(height2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            median_angle = np.median(angles)
            if abs(median_angle) < 45:  # Reasonable skew
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_REPLICATE)
        
        return img

    def detect_table_regions(self, img: np.ndarray) -> List[Dict]:
        """Enhanced table region detection with multiple strategies."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Strategy 1: Horizontal line detection (row-focused)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 2))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        
        # Strategy 2: Vertical line detection (column-focused)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 60))
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        
        # Strategy 3: Text block density
        text_blocks = cv2.dilate(h_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)))
        
        # Combine all strategies
        combined = cv2.addWeighted(h_lines, 0.4, v_lines, 0.4, 0)
        combined = cv2.addWeighted(combined, 0.7, text_blocks, 0.3, 0)
        
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        table_regions = []
        h_img, w_img = img.shape[:2]
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Filter criteria
            if (w > w_img * self.config.min_table_width_ratio and 
                h > self.config.min_table_height and 
                area > 2000 and w/h < 10):  # Reasonable aspect ratio
                
                table_regions.append({
                    'bbox': (x, y, x+w, y+h),
                    'area': area,
                    'confidence': min(1.0, h_lines[y:y+h, x:x+w].sum() / (w*h*255))
                })
        
        # Sort by area (largest first) and filter overlaps
        table_regions.sort(key=lambda t: t['area'], reverse=True)
        filtered = []
        
        for table in table_regions:
            if not any(self._bbox_iou(table['bbox'], f['bbox']) > 0.6 
                      for f in filtered):
                filtered.append(table)
        
        return filtered[:5]  # Limit to top 5 tables per page

    def _bbox_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Intersection over Union for bounding boxes."""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - inter
        return inter / union if union > 0 else 0.0

    def extract_words_with_multiple_psm(self, img: np.ndarray) -> List[Dict]:
        """Run OCR with multiple PSM modes and merge best results."""
        best_words = []
        best_confidence = 0
        
        for psm in self.config.psm_modes:
            try:
                config_str = f'--oem 3 --psm {psm} -l {self.config.ocr_languages}'
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, 
                                               config=config_str)
                
                words = []
                total_conf = 0
                word_count = 0
                
                for i in range(len(data['text'])):
                    if int(data['conf'][i]) > 10:  # Skip very low confidence
                        conf = float(data['conf'][i]) / 100.0
                        total_conf += conf
                        word_count += 1
                        
                        bbox = (data['left'][i], data['top'][i], 
                               data['left'][i] + data['width'][i],
                               data['top'][i] + data['height'][i])
                        
                        words.append({
                            'text': data['text'][i].strip(),
                            'bbox': bbox,
                            'confidence': conf
                        })
                
                avg_conf = total_conf / max(1, word_count)
                if avg_conf > best_confidence and words:
                    best_words = words
                    best_confidence = avg_conf
                    
            except Exception as e:
                logger.debug(f"PSM {psm} failed: {e}")
                continue
        
        return best_words

    def cluster_into_table(self, words: List[Dict], bbox: Tuple[int, int, int, int]) -> Table:
        """Intelligent word clustering into table structure."""
        x1, y1, x2, y2 = bbox
        table_words = [w for w in words if (x1 <= w['bbox'][0] <= x2 and 
                                          y1 <= w['bbox'][1] <= y2)]
        
        if len(table_words) < 3:
            return None
        
        # Sort by Y then X
        table_words.sort(key=lambda w: (w['bbox'][1], w['bbox'][0]))
        
        # Group into rows
        rows = []
        current_row = []
        current_y = table_words[0]['bbox'][1]
        
        for word in table_words:
            if abs(word['bbox'][1] - current_y) <= self.config.row_tolerance:
                current_row.append(word)
            else:
                if current_row:
                    rows.append(self._group_row_into_cells(current_row))
                current_row = [word]
                current_y = word['bbox'][1]
        
        if current_row:
            rows.append(self._group_row_into_cells(current_row))
        
        # Filter empty rows
        rows = [row for row in rows if any(cell.text.strip() for cell in row)]
        
        if len(rows) < 2:
            return None
        
        # Calculate confidence
        total_conf = sum(sum(cell.confidence or 0 for cell in row) for row in rows)
        avg_conf = total_conf / sum(len(row) for row in rows)
        
        return Table(bbox=bbox, rows=rows, confidence=avg_conf)

    def _group_row_into_cells(self, row_words: List[Dict]) -> List[TableCell]:
        """Group words in row into cells based on X-gaps."""
        if not row_words:
            return []
        
        row_words.sort(key=lambda w: w['bbox'][0])
        cells = []
        current_cell_words = [row_words[0]]
        last_x_end = row_words[0]['bbox'][2]
        
        for word in row_words[1:]:
            gap = word['bbox'][0] - last_x_end
            if gap > self.config.col_gap_threshold:
                # End current cell
                cell_text = ' '.join(w['text'] for w in current_cell_words)
                avg_conf = np.mean([w['confidence'] or 0 for w in current_cell_words])
                bbox = (min(w['bbox'][0] for w in current_cell_words),
                       min(w['bbox'][1] for w in current_cell_words),
                       max(w['bbox'][2] for w in current_cell_words),
                       max(w['bbox'][3] for w in current_cell_words))
                
                cells.append(TableCell(text=cell_text.strip(), bbox=bbox, confidence=avg_conf))
                current_cell_words = [word]
            else:
                current_cell_words.append(word)
            
            last_x_end = word['bbox'][2]
        
        # Add final cell
        if current_cell_words:
            cell_text = ' '.join(w['text'] for w in current_cell_words)
            avg_conf = np.mean([w['confidence'] or 0 for w in current_cell_words])
            bbox = (min(w['bbox'][0] for w in current_cell_words),
                   min(w['bbox'][1] for w in current_cell_words),
                   max(w['bbox'][2] for w in current_cell_words),
                   max(w['bbox'][3] for w in current_cell_words))
            
            cells.append(TableCell(text=cell_text.strip(), bbox=bbox, confidence=avg_conf))
        
        return cells

    def process_page(self, pdf_path: str, page_idx: int) -> PageContent:
        """Process single page with enhanced OCR and table extraction."""
        pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd
        
        doc = fitz.open(pdf_path)
        page = doc[page_idx]
        
        # Render page
        mat = fitz.Matrix(self.config.dpi / 72, self.config.dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Deskew for better detection
        img = self.deskew_image(img)
        
        # Extract all words
        all_words = self.extract_words_with_multiple_psm(img)
        
        # Detect table regions
        table_regions = self.detect_table_regions(img)
        tables = []
        used_word_indices = set()
        
        # Process tables (largest first)
        for region in table_regions:
            table = self.cluster_into_table(all_words, region['bbox'])
            if table and table.confidence > 0.1:
                # Mark used words
                for i, word in enumerate(all_words):
                    wx1, wy1, wx2, wy2 = word['bbox']
                    tx1, ty1, tx2, ty2 = table.bbox
                    if (tx1 <= wx1 <= tx2 and ty1 <= wy1 <= ty2):
                        used_word_indices.add(i)
                tables.append(table)
        
        # Extract non-table text
        remaining_words = [w for i, w in enumerate(all_words) if i not in used_word_indices]
        text_blocks = self._format_remaining_text(remaining_words)
        
        # Calculate page stats
        ocr_confidence = np.mean([w['confidence'] or 0 for w in all_words]) if all_words else 0
        
        doc.close()
        return PageContent(
            page_number=page_idx + 1,
            text_blocks=text_blocks,
            tables=tables,
            total_words=len(all_words),
            ocr_confidence=ocr_confidence
        )
    
    def _format_remaining_text(self, words: List[Dict]) -> List[str]:
        """Format remaining text into logical blocks."""
        if not words:
            return []
        
        words.sort(key=lambda w: (w['bbox'][1], w['bbox'][0]))
        blocks = []
        current_block = []
        current_y = words[0]['bbox'][1]
        
        for word in words:
            if abs(word['bbox'][1] - current_y) <= 12:
                current_block.append(word['text'])
            else:
                if current_block:
                    blocks.append(' '.join(current_block))
                current_block = [word['text']]
                current_y = word['bbox'][1]
        
        if current_block:
            blocks.append(' '.join(current_block))
        
        return [block.strip() for block in blocks if block.strip()]

def process_document(pdf_path: str, output_json: str = "structured_output.json", 
                    output_text: Optional[str] = None) -> List[PageContent]:
    """Process entire document with parallel processing."""
    config = OCRConfig()
    extractor = TableExtractor(config)
    
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()
    
    logger.info(f"Processing {num_pages} pages with {config.parallel_workers} workers...")
    
    args = [(pdf_path, i) for i in range(num_pages)]
    with Pool(config.parallel_workers) as pool:
        results = pool.starmap(extractor.process_page, args)
    
    # Save JSON output
    json_data = [asdict(page) for page in results]
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"JSON output saved to {output_json}")
    
    # Optional text output
    if output_text:
        with open(output_text, 'w', encoding='utf-8') as f:
            for page in results:
                f.write(f"=== PAGE {page.page_number} ===\n\n")
                if page.text_blocks:
                    f.write("TEXT:\n")
                    for block in page.text_blocks:
                        f.write(f"  {block}\n")
                
                for i, table in enumerate(page.tables):
                    f.write(f"\nTABLE {i+1} (confidence: {table.confidence:.2f}):\n")
                    for row_idx, row in enumerate(table.rows):
                        row_text = [cell.text for cell in row]
                        f.write(f"  {row_idx+1}: {' | '.join(row_text)}\n")
                
                f.write("\n" + "="*50 + "\n")
    
    return results

if __name__ == "__main__":
    pdf_file = "scanneddoc/Two Wheeler Product Process Note 1.pdf"
    results = process_document(pdf_file, "structured_output.json", "readable_output.txt")
    print("Extraction complete! Check structured_output.json and readable_output.txt")
