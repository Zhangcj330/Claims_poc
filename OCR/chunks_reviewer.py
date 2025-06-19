import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from dataclasses import dataclass
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChunkReviewResult:
    """Results of chunk review"""
    page_no: int
    is_accurate: bool
    confidence_score: float
    issues_found: List[str]
    improved_chunks: Optional[List[Dict[str, Any]]]
    review_summary: str

class ChunksReviewer:
    """
    OCR Chunks Review System using Gemini 2.5
    Reviews existing chunks against original PDF images and generates improved versions
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the reviewer with Gemini API"""
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            api_key=self.api_key,
            temperature=0.1,
        )
        
        # Review prompt for accuracy checking
        self.review_prompt = """
You are an expert document reviewer specializing in insurance Product Disclosure Statements (PDS).
Your task is to review OCR-extracted chunks against the original PDF page image to assess accuracy and completeness.

REVIEW CRITERIA:
1. **Text Accuracy**: Compare extracted text with image content for accuracy
2. **Structure Preservation**: Check if section titles, headings, and hierarchy are correctly captured
3. **Content Completeness**: Ensure no important content is missing or truncated
4. **Metadata Accuracy**: Verify Section_Title, Subheading, and content_label assignments
5. **Logical Chunking**: Assess if chunks represent coherent semantic units

ANALYSIS PROCESS:
1. Carefully examine the PDF page image
2. Compare it with the provided chunks data
3. Identify discrepancies, missing content, or structural issues
4. Rate accuracy on a scale of 0-100
5. Provide specific improvement recommendations

Return a JSON response with this EXACT structure:
{
  "is_accurate": boolean,
  "confidence_score": number (0-100),
  "issues_found": [
    "string describing specific issues"
  ],
  "review_summary": "string with overall assessment",
  "needs_regeneration": boolean
}

ACCURACY THRESHOLDS:
- 90-100: Excellent, minor or no issues
- 70-89: Good, some improvements needed
- 50-69: Moderate issues, significant improvements required
- 0-49: Poor accuracy, complete regeneration recommended

Focus on practical issues that would impact document understanding and information retrieval.
"""

        # Regeneration prompt for improved chunks
        self.regeneration_prompt = """
You are an expert document processing AI specializing in insurance Product Disclosure Statements (PDS).
Based on the review feedback, regenerate improved chunks for this page that address the identified issues.

IMPROVEMENT REQUIREMENTS:
1. Fix all accuracy issues identified in the review
2. Ensure proper text extraction and OCR corrections
3. Maintain logical chunking (50-1000 words per chunk)
4. Preserve document structure and hierarchy
5. Assign correct metadata (Section_Title, Subheading, content_label)

Return a JSON array of improved chunks with this EXACT schema:
[
  {
    "Section_Title": "string",
    "Subheading": "string", 
    "content": "string",
    "content_label": "string"
  }
]

CHUNKING RULES:
- Each chunk should be a coherent semantic unit
- Respect natural paragraph and section boundaries
- Use proper content_label: "text", "image", "table", "list", "equation", "figure"
- For Section_Title: Only specify if this page starts a new major section
- For Subheading: Use immediate subsection heading relevant to the chunk
- Use "N/A" when no specific title/heading applies

Ensure the improved chunks accurately reflect the content visible in the PDF page image.
"""

    def load_chunks_from_jsonl(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """Load chunks from JSONL file"""
        chunks = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
            logger.info(f"Loaded {len(chunks)} chunks from {jsonl_path}")
            return chunks
        except Exception as e:
            logger.error(f"Error loading chunks from {jsonl_path}: {e}")
            return []

    def group_chunks_by_page(self, chunks: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Group chunks by page number"""
        page_groups = {}
        for chunk in chunks:
            page_no = chunk.get('Page_no', 1)
            if page_no not in page_groups:
                page_groups[page_no] = []
            page_groups[page_no].append(chunk)
        
        logger.info(f"Grouped chunks into {len(page_groups)} pages")
        return page_groups

    def pdf_page_to_image(self, pdf_path: str, page_no: int, dpi: int = 200) -> Optional[Image.Image]:
        """Convert specific PDF page to image"""
        try:
            doc = fitz.open(pdf_path)
            if page_no < 1 or page_no > len(doc):
                logger.error(f"Page {page_no} not found in PDF (total pages: {len(doc)})")
                return None
            
            page = doc.load_page(page_no - 1)  # PDF pages are 0-indexed
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            img = Image.open(io.BytesIO(img_data))
            doc.close()
            return img
        except Exception as e:
            logger.error(f"Error converting page {page_no} to image: {e}")
            return None

    def encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def review_page_chunks(self, page_image: Image.Image, page_chunks: List[Dict[str, Any]], 
                          page_no: int) -> ChunkReviewResult:
        """Review chunks for a specific page against the original image"""
        try:
            # Prepare chunks data for review
            chunks_summary = []
            for i, chunk in enumerate(page_chunks):
                chunks_summary.append({
                    "chunk_id": i + 1,
                    "Section_Title": chunk.get('Section_Title', 'N/A'),
                    "Subheading": chunk.get('Subheading', 'N/A'),
                    "content_label": chunk.get('content_label', 'text'),
                    "content_preview": chunk.get('content', '')[:200] + "..." if len(chunk.get('content', '')) > 200 else chunk.get('content', ''),
                    "content_length": len(chunk.get('content', ''))
                })

            # Create review prompt
            review_prompt = f"""
{self.review_prompt}

PAGE INFORMATION:
- Page Number: {page_no}
- Number of chunks: {len(page_chunks)}

CURRENT CHUNKS DATA:
{json.dumps(chunks_summary, indent=2, ensure_ascii=False)}

Please review these chunks against the PDF page image and provide your assessment.
"""

            # Convert image to base64
            image_base64 = self.encode_image(page_image)
            
            # Create message with image and text
            content_parts = [
                {"type": "text", "text": review_prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
            ]
            
            message = HumanMessage(content=content_parts)
            response = self.model.invoke([message])
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if not json_match:
                logger.error(f"Could not parse review response for page {page_no}")
                return ChunkReviewResult(
                    page_no=page_no,
                    is_accurate=False,
                    confidence_score=0,
                    issues_found=["Failed to parse review response"],
                    improved_chunks=None,
                    review_summary="Review parsing failed"
                )
            
            review_data = json.loads(json_match.group())
            
            return ChunkReviewResult(
                page_no=page_no,
                is_accurate=review_data.get('is_accurate', False),
                confidence_score=review_data.get('confidence_score', 0),
                issues_found=review_data.get('issues_found', []),
                improved_chunks=None,  # Will be filled if regeneration is needed
                review_summary=review_data.get('review_summary', '')
            )
            
        except Exception as e:
            logger.error(f"Error reviewing page {page_no}: {e}")
            return ChunkReviewResult(
                page_no=page_no,
                is_accurate=False,
                confidence_score=0,
                issues_found=[f"Review error: {str(e)}"],
                improved_chunks=None,
                review_summary="Review failed due to error"
            )

    def regenerate_chunks(self, page_image: Image.Image, page_no: int, 
                         review_result: ChunkReviewResult, 
                         global_metadata: Dict[str, str]) -> List[Dict[str, Any]]:
        """Generate improved chunks based on review feedback"""
        try:
            # Prepare regeneration prompt with feedback
            regeneration_prompt = f"""
{self.regeneration_prompt}

PAGE INFORMATION:
- Page Number: {page_no}
- Document: {global_metadata.get('Document_Name', 'Unknown')}
- Insurer: {global_metadata.get('Insurer', 'Unknown')}

REVIEW FEEDBACK:
- Confidence Score: {review_result.confidence_score}
- Issues Found: {', '.join(review_result.issues_found)}
- Review Summary: {review_result.review_summary}

Based on this feedback and the PDF page image, generate improved chunks that address all identified issues.
"""

            # Convert image to base64
            image_base64 = self.encode_image(page_image)
            
            # Create message with image and text
            content_parts = [
                {"type": "text", "text": regeneration_prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
            ]
            
            message = HumanMessage(content=content_parts)
            response = self.model.invoke([message])
            
            # Parse JSON response
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if not json_match:
                logger.error(f"Could not parse regeneration response for page {page_no}")
                return []
            
            improved_chunks = json.loads(json_match.group())
            
            # Add global metadata to each chunk
            for chunk in improved_chunks:
                chunk.update({
                    'Insurer': global_metadata.get('Insurer', 'Unknown'),
                    'Document_Name': global_metadata.get('Document_Name', 'Unknown'),
                    'Document_Date': global_metadata.get('Document_Date', 'Unknown'),
                    'Product_type': global_metadata.get('Product_type', 'Unknown'),
                    'Page_no': page_no
                })
            
            logger.info(f"Generated {len(improved_chunks)} improved chunks for page {page_no}")
            return improved_chunks
            
        except Exception as e:
            logger.error(f"Error regenerating chunks for page {page_no}: {e}")
            return []

    def extract_global_metadata(self, chunks: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract global metadata from existing chunks"""
        if not chunks:
            return {}
        
        first_chunk = chunks[0]
        return {
            'Insurer': first_chunk.get('Insurer', 'Unknown'),
            'Document_Name': first_chunk.get('Document_Name', 'Unknown'),
            'Document_Date': first_chunk.get('Document_Date', 'Unknown'),
            'Product_type': first_chunk.get('Product_type', 'Unknown')
        }

    def review_and_improve_chunks(self, jsonl_path: str, pdf_path: str, 
                                 output_dir: Optional[str] = None,
                                 confidence_threshold: float = 75.0) -> Dict[str, Any]:
        """
        Main method to review and improve chunks
        
        Args:
            jsonl_path: Path to existing chunks JSONL file
            pdf_path: Path to original PDF file
            output_dir: Output directory for improved chunks
            confidence_threshold: Minimum confidence score to accept chunks
            
        Returns:
            Dict with review results and improved chunks
        """
        logger.info(f"Starting chunk review for {jsonl_path}")
        
        # Load existing chunks
        chunks = self.load_chunks_from_jsonl(jsonl_path)
        if not chunks:
            return {"error": "No chunks loaded"}
        
        # Extract global metadata
        global_metadata = self.extract_global_metadata(chunks)
        
        # Group chunks by page
        page_groups = self.group_chunks_by_page(chunks)
        
        # Process each page
        review_results = []
        improved_chunks_data = []
        total_pages = len(page_groups)
        pages_improved = 0
        
        for page_no in sorted(page_groups.keys()):
            logger.info(f"Processing page {page_no}/{total_pages}")
            
            # Get page image
            page_image = self.pdf_page_to_image(pdf_path, page_no)
            if not page_image:
                logger.warning(f"Skipping page {page_no} - could not load image")
                continue
            
            # Review chunks for this page
            page_chunks = page_groups[page_no]
            review_result = self.review_page_chunks(page_image, page_chunks, page_no)
            review_results.append(review_result)
            
            # Decide whether to regenerate chunks
            needs_improvement = (
                not review_result.is_accurate or 
                review_result.confidence_score < confidence_threshold
            )
            
            if needs_improvement:
                logger.info(f"Page {page_no} needs improvement (confidence: {review_result.confidence_score})")
                improved_chunks = self.regenerate_chunks(
                    page_image, page_no, review_result, global_metadata
                )
                
                if improved_chunks:
                    review_result.improved_chunks = improved_chunks
                    improved_chunks_data.extend(improved_chunks)
                    pages_improved += 1
                else:
                    # Keep original chunks if regeneration failed
                    improved_chunks_data.extend(page_chunks)
            else:
                logger.info(f"Page {page_no} is accurate (confidence: {review_result.confidence_score})")
                # Keep original chunks
                improved_chunks_data.extend(page_chunks)
        
        # Save results
        if output_dir is None:
            output_dir = os.path.dirname(jsonl_path)
        
        # Generate output filenames
        base_name = Path(jsonl_path).stem.replace('_chunks', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save improved chunks
        improved_jsonl_path = os.path.join(output_dir, f"{base_name}_chunks_improved_{timestamp}.jsonl")
        with open(improved_jsonl_path, 'w', encoding='utf-8') as f:
            for chunk in improved_chunks_data:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        # Save review report
        review_report_path = os.path.join(output_dir, f"{base_name}_review_report_{timestamp}.json")
        review_report = {
            "review_summary": {
                "total_pages": total_pages,
                "pages_improved": pages_improved,
                "improvement_rate": pages_improved / total_pages if total_pages > 0 else 0,
                "average_confidence": sum(r.confidence_score for r in review_results) / len(review_results) if review_results else 0,
                "confidence_threshold": confidence_threshold,
                "timestamp": timestamp
            },
            "page_reviews": [
                {
                    "page_no": r.page_no,
                    "is_accurate": r.is_accurate,
                    "confidence_score": r.confidence_score,
                    "issues_found": r.issues_found,
                    "review_summary": r.review_summary,
                    "was_improved": r.improved_chunks is not None
                }
                for r in review_results
            ]
        }
        
        with open(review_report_path, 'w', encoding='utf-8') as f:
            json.dump(review_report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        logger.info(f"\nReview completed!")
        logger.info(f"├── Total pages processed: {total_pages}")
        logger.info(f"├── Pages improved: {pages_improved}")
        logger.info(f"├── Improvement rate: {pages_improved/total_pages*100:.1f}%")
        logger.info(f"├── Average confidence: {review_report['review_summary']['average_confidence']:.1f}")
        logger.info(f"├── Improved chunks: {improved_jsonl_path}")
        logger.info(f"└── Review report: {review_report_path}")
        
        return {
            "review_results": review_results,
            "improved_chunks": improved_chunks_data,
            "files": {
                "improved_chunks": improved_jsonl_path,
                "review_report": review_report_path
            },
            "summary": review_report["review_summary"]
        }

def main():
    """Example usage"""
    reviewer = ChunksReviewer()
    
    # Example files - adjust paths as needed
    jsonl_file = "ocr/Data_Json/TAL_AcceleratedProtection_2022-08-05_chunks.jsonl"
    pdf_file = "ocr/TAL_AcceleratedProtection_2022-08-05.pdf"
    
    if os.path.exists(jsonl_file) and os.path.exists(pdf_file):
        results = reviewer.review_and_improve_chunks(
            jsonl_path=jsonl_file,
            pdf_path=pdf_file,
            confidence_threshold=75.0  # Adjust as needed
        )
        
        print(f"\nReview completed with {results['summary']['improvement_rate']*100:.1f}% improvement rate")
    else:
        print(f"Files not found:")
        print(f"JSONL: {jsonl_file} (exists: {os.path.exists(jsonl_file)})")
        print(f"PDF: {pdf_file} (exists: {os.path.exists(pdf_file)})")

if __name__ == "__main__":
    main() 