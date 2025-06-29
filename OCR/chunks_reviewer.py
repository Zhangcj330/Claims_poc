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

    def display_review_results(self, review_results: List[ChunkReviewResult]) -> None:
        """Display review results in a formatted way"""
        print("\n" + "="*80)
        print("📋 CHUNKS REVIEW RESULTS")
        print("="*80)
        
        total_pages = len(review_results)
        low_confidence_pages = [r for r in review_results if r.confidence_score < 75]
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"   Total pages reviewed: {total_pages}")
        print(f"   Low confidence pages: {len(low_confidence_pages)}")
        print(f"   Average confidence: {sum(r.confidence_score for r in review_results) / len(review_results):.1f}")
        
        # Detailed results
        print(f"\n📄 DETAILED RESULTS:")
        for result in review_results:
            status_icon = "✅" if result.confidence_score >= 75 else "⚠️" if result.confidence_score >= 50 else "❌"
            print(f"\n{status_icon} Page {result.page_no}: {result.confidence_score:.1f}% confidence")
            print(f"   Summary: {result.review_summary}")
            
            if result.issues_found:
                print(f"   Issues found:")
                for issue in result.issues_found:
                    print(f"     • {issue}")

    def get_user_selection(self, review_results: List[ChunkReviewResult]) -> List[int]:
        """Get user selection for which pages to regenerate"""
        print("\n" + "="*80)
        print("🤔 MANUAL SELECTION - Choose pages to regenerate")
        print("="*80)
        
        # Show selectable pages
        selectable_pages = []
        print("\nPages available for regeneration:")
        for result in review_results:
            if not result.is_accurate or result.confidence_score < 90:
                selectable_pages.append(result.page_no)
                confidence_icon = "🔴" if result.confidence_score < 50 else "🟡" if result.confidence_score < 75 else "🟢"
                print(f"  {confidence_icon} Page {result.page_no}: {result.confidence_score:.1f}% - {result.review_summary[:60]}...")
        
        if not selectable_pages:
            print("✨ All pages have high confidence scores. No regeneration needed!")
            return []
        
        print(f"\nAvailable options:")
        print(f"  • Enter page numbers (comma-separated, e.g., '1,3,5')")
        print(f"  • Enter 'all' to regenerate all problematic pages")
        print(f"  • Enter 'none' or press Enter to skip regeneration")
        print(f"  • Enter 'auto' to use automatic threshold (confidence < 75)")
        
        while True:
            try:
                user_input = input(f"\n👤 Your choice: ").strip().lower()
                
                if user_input == '' or user_input == 'none':
                    print("✋ Skipping regeneration. Original chunks will be kept.")
                    return []
                
                elif user_input == 'all':
                    print(f"🔄 Will regenerate all {len(selectable_pages)} problematic pages.")
                    return selectable_pages
                
                elif user_input == 'auto':
                    auto_pages = [r.page_no for r in review_results if r.confidence_score < 75]
                    print(f"🤖 Using automatic selection: {len(auto_pages)} pages with confidence < 75%")
                    return auto_pages
                
                else:
                    # Parse comma-separated page numbers
                    page_numbers = [int(p.strip()) for p in user_input.split(',')]
                    
                    # Validate page numbers
                    invalid_pages = [p for p in page_numbers if p not in selectable_pages]
                    if invalid_pages:
                        print(f"❌ Invalid page numbers: {invalid_pages}. Available pages: {selectable_pages}")
                        continue
                    
                    print(f"✅ Will regenerate {len(page_numbers)} selected pages: {page_numbers}")
                    return page_numbers
                    
            except ValueError:
                print("❌ Invalid input. Please enter page numbers, 'all', 'none', or 'auto'.")
            except KeyboardInterrupt:
                print("\n\n👋 Operation cancelled by user.")
                return []

    def confirm_regeneration(self, selected_pages: List[int]) -> bool:
        """Ask for final confirmation before regeneration"""
        if not selected_pages:
            return False
            
        print(f"\n⚡ About to regenerate {len(selected_pages)} pages: {selected_pages}")
        print("⚠️  This will use API calls and may take some time.")
        
        while True:
            try:
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    return True
                elif confirm in ['n', 'no']:
                    print("❌ Regeneration cancelled.")
                    return False
                else:
                    print("Please enter 'y' or 'n'")
            except KeyboardInterrupt:
                print("\n\n👋 Operation cancelled by user.")
                return False

    def review_and_improve_chunks(self, jsonl_path: str, pdf_path: str, 
                                 output_dir: Optional[str] = None,
                                 confidence_threshold: float = 75.0,
                                 interactive: bool = True) -> Dict[str, Any]:
        """
        Main method to review and improve chunks
        
        Args:
            jsonl_path: Path to existing chunks JSONL file
            pdf_path: Path to original PDF file
            output_dir: Output directory for improved chunks
            confidence_threshold: Minimum confidence score to accept chunks (used in auto mode)
            interactive: If True, ask user for manual selection. If False, use automatic threshold
            
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
        
        # PHASE 1: Review all pages
        print(f"\n🔍 PHASE 1: Reviewing {len(page_groups)} pages...")
        review_results = []
        
        for page_no in sorted(page_groups.keys()):
            logger.info(f"Reviewing page {page_no}/{len(page_groups)}")
            
            # Get page image
            page_image = self.pdf_page_to_image(pdf_path, page_no)
            if not page_image:
                logger.warning(f"Skipping page {page_no} - could not load image")
                continue
            
            # Review chunks for this page
            page_chunks = page_groups[page_no]
            review_result = self.review_page_chunks(page_image, page_chunks, page_no)
            review_results.append(review_result)
        
        # Display review results
        self.display_review_results(review_results)
        
        # PHASE 2: Get user selection or use automatic threshold
        if interactive:
            selected_pages = self.get_user_selection(review_results)
            if selected_pages and not self.confirm_regeneration(selected_pages):
                selected_pages = []
        else:
            # Automatic mode: use confidence threshold
            selected_pages = [r.page_no for r in review_results 
                            if not r.is_accurate or r.confidence_score < confidence_threshold]
            print(f"\n🤖 Automatic mode: Selected {len(selected_pages)} pages for regeneration (confidence < {confidence_threshold})")
        
        # PHASE 3: Regenerate selected pages
        improved_chunks_data = []
        pages_improved = 0
        
        if selected_pages:
            print(f"\n🔄 PHASE 3: Regenerating {len(selected_pages)} selected pages...")
            
            for page_no in sorted(page_groups.keys()):
                page_chunks = page_groups[page_no]
                
                if page_no in selected_pages:
                    logger.info(f"Regenerating page {page_no}")
                    
                    # Get page image and review result
                    page_image = self.pdf_page_to_image(pdf_path, page_no)
                    review_result = next((r for r in review_results if r.page_no == page_no), None)
                    
                    if page_image and review_result:
                        improved_chunks = self.regenerate_chunks(
                            page_image, page_no, review_result, global_metadata
                        )
                        
                        if improved_chunks:
                            improved_chunks_data.extend(improved_chunks)
                            pages_improved += 1
                            
                            # Update review result with improved chunks
                            review_result.improved_chunks = improved_chunks
                        else:
                            # Keep original chunks if regeneration failed
                            improved_chunks_data.extend(page_chunks)
                    else:
                        # Keep original chunks if image/review not available
                        improved_chunks_data.extend(page_chunks)
                else:
                    # Keep original chunks for non-selected pages
                    improved_chunks_data.extend(page_chunks)
        else:
            print(f"\n✋ No pages selected for regeneration. Keeping original chunks.")
            # Keep all original chunks
            for page_chunks in page_groups.values():
                improved_chunks_data.extend(page_chunks)
        
        # PHASE 4: Save results
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
                "total_pages": len(page_groups),
                "pages_selected_for_improvement": len(selected_pages),
                "pages_actually_improved": pages_improved,
                "improvement_rate": pages_improved / len(page_groups) if len(page_groups) > 0 else 0,
                "average_confidence": sum(r.confidence_score for r in review_results) / len(review_results) if review_results else 0,
                "confidence_threshold": confidence_threshold,
                "interactive_mode": interactive,
                "selected_pages": selected_pages,
                "timestamp": timestamp
            },
            "page_reviews": [
                {
                    "page_no": r.page_no,
                    "is_accurate": r.is_accurate,
                    "confidence_score": r.confidence_score,
                    "issues_found": r.issues_found,
                    "review_summary": r.review_summary,
                    "was_selected_for_improvement": r.page_no in selected_pages,
                    "was_actually_improved": r.improved_chunks is not None
                }
                for r in review_results
            ]
        }
        
        with open(review_report_path, 'w', encoding='utf-8') as f:
            json.dump(review_report, f, indent=2, ensure_ascii=False)
        
        # Print final summary
        print(f"\n" + "="*80)
        print(f"✅ REVIEW COMPLETED!")
        print(f"="*80)
        print(f"📊 Total pages processed: {len(page_groups)}")
        print(f"🎯 Pages selected for improvement: {len(selected_pages)}")
        print(f"🔄 Pages actually improved: {pages_improved}")
        print(f"📈 Improvement rate: {pages_improved/len(page_groups)*100:.1f}%")
        print(f"🎯 Average confidence: {review_report['review_summary']['average_confidence']:.1f}")
        print(f"📁 Improved chunks: {improved_jsonl_path}")
        print(f"📋 Review report: {review_report_path}")
        print(f"="*80)
        
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
    """Example usage with interactive mode"""
    reviewer = ChunksReviewer()
    
    # Example files - adjust paths as needed
    jsonl_file = "Data_Json/TAL_AcceleratedProtection_2022-08-05_chunks.jsonl"
    pdf_file = "TAL_AcceleratedProtection_2022-08-05.pdf"
    
    if os.path.exists(jsonl_file) and os.path.exists(pdf_file):
        print("🚀 Starting interactive chunk review...")
        print("💡 You'll be able to manually select which pages to regenerate after the review phase.")
        
        results = reviewer.review_and_improve_chunks(
            jsonl_path=jsonl_file,
            pdf_path=pdf_file,
            confidence_threshold=75.0,
            interactive=True  # Enable human-in-the-loop
        )
        
        print(f"\n🎉 Review completed! Check the output files for results.")
    else:
        print(f"❌ Files not found:")
        print(f"JSONL: {jsonl_file} (exists: {os.path.exists(jsonl_file)})")
        print(f"PDF: {pdf_file} (exists: {os.path.exists(pdf_file)})")

if __name__ == "__main__":
    main() 