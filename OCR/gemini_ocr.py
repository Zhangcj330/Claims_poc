import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GeminiPDSProcessor:
    """
    PDS OCR & Chunking System using Gemini 2.5 Flash for intelligent document processing
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the processor with Gemini API"""
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            api_key=self.api_key,
            temperature=0.1,
        )
        
        # System prompt for document understanding and chunking
        self.system_prompt = """
You are an expert document processing AI specializing in insurance Product Disclosure Statements (PDS). 
Your task is to analyze each page image and extract structured content with precise metadata.

CRITICAL REQUIREMENTS:
1. Extract coherent chunks based on document structure (headings, sections, paragraphs)
2. Each chunk should be 50-1000 words and represent a logical unit of content
3. Preserve hierarchical relationships (sections, subsections)
4. Focus on PAGE-LEVEL metadata extraction only
5. Use heading detection to determine section structure
6. Handle tables, lists, and complex layouts intelligently
7. **CONTEXT AWARENESS**: Consider previous page context for section continuity

For each page, return a JSON array of chunks following this EXACT schema:
{
  "Section_Title": "string",
  "Subheading": "string", 
  "content": "string",
  "content_label": "string"
}

PAGE-LEVEL METADATA EXTRACTION RULES:
- Section_Title: Top-level section heading on this page (e.g., "Section 2: Benefits", "Claims", "Definitions")
  * If NO new section heading is visible on this page, use "N/A" - the system will inherit from previous page
  * Only specify a Section_Title if you see a clear, new section heading on this page
- Subheading: Immediate subsection heading for this chunk (e.g., "Life Insurance Benefits", "Exclusions")
  * Same inheritance logic applies - use "N/A" if no new subheading is found
- content: Main body text of the chunk (50-1000 words, coherent semantic unit)
- content_label: Label for the content type (e.g., "text", "image", "table", "list", "equation", "figure")

SECTION CONTINUITY RULES:
- Many sections span multiple pages in insurance documents
- Only change Section_Title when you see a NEW section heading (e.g., "Section 3:", "Claims Process", "Definitions")
- If the page is continuing content from a previous section without a new heading, use "N/A" for Section_Title
- The system will automatically inherit the correct Section_Title from the previous page
- This ensures consistent metadata across pages within the same logical section

IMPORTANT NOTES:
- DO NOT extract global document metadata (Insurer, Document_Name, Document_Date, Product_type, Page_no)
- Focus ONLY on page-specific content structure and semantic chunking
- Ensure each chunk represents a logical, self-contained unit of information
- Respect document hierarchy: group content under appropriate section/subsection headings
- Pay attention to visual cues like checkmarks, bullet points, and formatting

CRITICAL: If the page contains NO substantial text content (e.g., cover pages, pure image pages, blank pages), return a single chunk with content_label: "image" and content describing what is visible on the page (e.g., "Cover page with company logo", "Blank page", "Page contains only images/diagrams"). This ensures every page generates a chunk for tracking purposes.
"""

    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[tuple]:
        """Convert PDF pages to images with page numbers"""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Convert to image
            mat = fitz.Matrix(dpi/72, dpi/72)  # zoom factor
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            images.append((img, page_num + 1))
        
        doc.close()
        return images

    def encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def extract_global_metadata(self, pdf_path: str) -> Dict[str, str]:
        """Extract document-level metadata that applies to all chunks"""
        doc = fitz.open(pdf_path)
        metadata = {
            "document_name": Path(pdf_path).stem,
            "insurer": "Unknown",
            "document_date": "Unknown",
            "product_type": "Unknown"
        }
        
        # Try to extract from PDF metadata
        doc_metadata = doc.metadata
        if doc_metadata.get('title'):
            metadata["document_name"] = doc_metadata['title']
        
        # Extract from first few pages using OCR
        try:
            first_pages = []
            for page_num in range(min(5, len(doc))):  # First 3 pages
                page = doc.load_page(page_num)
                mat = fitz.Matrix(2, 2)  # Higher resolution for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                first_pages.append(img)
            
            # Use Gemini to extract metadata from first pages
            if first_pages:
                global_metadata = self._extract_document_metadata(first_pages, pdf_path)
                metadata.update(global_metadata)
                
        except Exception as e:
            print(f"Warning: Could not extract global metadata: {e}")
        
        doc.close()
        return metadata

    def _extract_document_metadata(self, images: List[Image.Image], file_name: str) -> Dict[str, str]:
        """Use Gemini to extract document-level metadata from first few pages"""
        prompt = f"""
Analyze the first few pages of an insurance document, along with the file name {file_name}, to extract the following information:
1. Insurer name (company providing the insurance)
2. Document name/title
3. Document date (publication or effective date in YYYY-MM-DD format)
4. Primary product type (Life, TPD, Trauma, Income Protection, etc.)

Return only a JSON object with these keys: insurer, document_name, document_date, product_type
If any information is not found, use "Unknown".
"""
        
        try:
            # Convert PIL images to base64 for LangChain
            content_parts = [{"type": "text", "text": prompt}]
            
            for img in images[:2]:  # Limit to first 2 pages for metadata
                # Convert PIL Image to base64
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                content_parts.append({
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{image_base64}"
                })
            
            # Create HumanMessage with multimodal content
            message = HumanMessage(content=content_parts)
            
            response = self.model.invoke([message])
            
            # Parse JSON response
            json_match = re.search(r'\{[^}]+\}', response.content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"Warning: Could not extract metadata with Gemini: {e}")
        
        return {}

    def process_page(self, image: Image.Image, page_num: int, global_metadata: Dict[str, str], previous_context: Optional[Dict[str, str]] = None, max_retries: int = 3) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Process a single page and extract chunks with retry mechanism, returning chunks and current context"""
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Prepare context information for the prompt
                context_info = ""
                if previous_context:
                    context_info = f"""
CONTEXT FROM PREVIOUS PAGE:
- Previous Section_Title: "{previous_context.get('Section_Title', 'N/A')}"
- Previous Subheading: "{previous_context.get('Subheading', 'N/A')}"    

CONTEXT RULES:
- If this page does not contain a new Section_Title heading, inherit the previous Section_Title
- Only change Section_Title if you see a clear new section heading on this page
- This ensures consistency across pages within the same section
"""
                
                # Convert PIL Image to base64 for LangChain
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                # Create human message content with text and image
                human_content = [
                    {
                        "type": "text", 
                        "text": f"""{context_info}

Process this page {page_num} of an insurance PDS document.
Global document metadata: {json.dumps(global_metadata)}

Extract chunks following the specified JSON schema."""
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{image_base64}"
                    }
                ]
                
                # Create messages with proper separation
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=human_content)
                ]
                
                response = self.model.invoke(messages)
                
                # Extract JSON from response
                chunks = self._parse_chunks_response(response.content, page_num, global_metadata, previous_context)
                
                # Check if chunks are valid (not error or fallback chunks that indicate failure)
                has_error_chunks = any(chunk.get('content_label') in ['error', 'fallback'] for chunk in chunks)
                
                if has_error_chunks and attempt < max_retries - 1:
                    print(f"    🔄 Attempt {attempt + 1} failed for page {page_num} (got error/fallback chunks), retrying...")
                    continue
                
                # Success or final attempt - extract current page context for next page
                current_context = self._extract_current_context(chunks, previous_context)
                
                if attempt > 0:
                    print(f"    ✅ Page {page_num} succeeded on attempt {attempt + 1}")
                
                return chunks, current_context
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"    🔄 Attempt {attempt + 1} failed for page {page_num}: {e}, retrying...")
                    continue
                else:
                    print(f"    ❌ All {max_retries} attempts failed for page {page_num}: {e}")
                    break
        
        # All retries failed, create error chunk
        error_msg = str(last_error) if last_error else "Unknown error after retries"
        error_chunks = self._create_failure_chunk(
            page_num, 
            global_metadata, 
            f"Error processing page {page_num}: Failed after {max_retries} attempts: {error_msg}",
            failure_type="error"
        )
        return error_chunks, previous_context or {}

    def _extract_current_context(self, chunks: List[Dict[str, Any]], previous_context: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Extract context information from current page chunks for next page"""
        if not chunks:
            return previous_context or {}
        
        # Find the most recent non-N/A Section_Title and Subheading
        current_context = previous_context.copy() if previous_context else {}
        
        for chunk in chunks:
            if chunk.get('Section_Title') and chunk['Section_Title'] != 'N/A':
                current_context['Section_Title'] = chunk['Section_Title']
            if chunk.get('Subheading') and chunk['Subheading'] != 'N/A':
                current_context['Subheading'] = chunk['Subheading']
        
        return current_context

    def _parse_chunks_response(self, response_text: str, page_num: int, global_metadata: Dict[str, str], previous_context: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Parse Gemini response and extract chunks with page-level metadata only"""
        chunks = []
        
        print(f"    📝 Parsing response for page {page_num}...")
        
        # Try to find JSON arrays in the response
        json_pattern = r'\[.*?\]'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        if not json_matches:
            # Try to find individual JSON objects with content field
            json_pattern = r'\{[^}]*"content"[^}]*\}'
            json_matches = re.findall(json_pattern, response_text, re.DOTALL)
            if json_matches:
                json_matches = ['[' + ','.join(json_matches) + ']']
        
        for json_str in json_matches:
            try:
                parsed_chunks = json.loads(json_str)
                if isinstance(parsed_chunks, list):
                    for chunk in parsed_chunks:
                        if isinstance(chunk, dict) and 'content' in chunk:
                            # Validate and clean chunk (adds global metadata consistently)
                            cleaned_chunk = self._validate_chunk(chunk, page_num, global_metadata, previous_context)
                            if cleaned_chunk:
                                chunks.append(cleaned_chunk)
                                print(f"      ✅ Valid chunk: Section='{chunk.get('Section_Title', 'N/A')}', Words={len(chunk['content'].split())}")
                break  # Use first valid JSON array
            except json.JSONDecodeError as e:
                print(f"    ⚠️ JSON decode error: {e}")
                continue
        
        # If no valid JSON found, create fallback chunk
        if not chunks:
            print(f"    ⚠️ No valid JSON found, creating fallback chunk")
            # Clean up the response text
            content = re.sub(r'[^\w\s.,;:!?()-]', ' ', response_text)
            content = re.sub(r'\s+', ' ', content).strip()
            
            if len(content) < 50:  # Too short to be useful
                content = f"Unable to process page {page_num} content properly."
            
            chunks = self._create_failure_chunk(
                page_num, 
                global_metadata, 
                content,
                failure_type="fallback",
                previous_context=previous_context
            )
        
        print(f"    📊 Generated {len(chunks)} chunks for page {page_num}")
        return chunks

    def _validate_chunk(self, chunk: Dict[str, Any], page_num: int, global_metadata: Dict[str, str], previous_context: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Validate and standardize chunk format, adding consistent global metadata and handling context inheritance"""
        
        # Define expected page-level fields from LLM
        page_level_fields = ["Section_Title", "Subheading", "content"]
        
        # Ensure all page-level fields exist
        for field in page_level_fields:
            if field not in chunk:
                chunk[field] = "N/A"
        
        # Handle context inheritance
        if previous_context:
            # If current chunk has no Section_Title or it's N/A, inherit from previous context
            if chunk.get("Section_Title") in ["N/A", "", None]:
                chunk["Section_Title"] = previous_context.get("Section_Title", "N/A")
                print(f"      🔗 Inherited Section_Title: '{chunk['Section_Title']}'")
                
                # Only inherit Subheading if we're continuing in the same section
                if chunk.get("Subheading") in ["N/A", "", None] and previous_context.get("Subheading") not in ["N/A", "", None]:
                    chunk["Subheading"] = previous_context.get("Subheading", "N/A")
                    print(f"      🔗 Inherited Subheading: '{chunk['Subheading']}'")
            
            # If we have a new Section_Title, don't inherit Subheading (new section starts fresh)
            elif chunk.get("Section_Title") not in ["N/A", "", None]:
                print(f"      🆕 New section detected: '{chunk['Section_Title']}' - not inheriting Subheading")
        
        # Validate content length (50-1000 words, except for image content)
        content = str(chunk.get("content", "")).strip()
        content_label = chunk.get("content_label", "text")
        
        if not content:  
            return None

        
        # Truncate if too long (keep within reasonable limits)
        words = content.split()
        if len(words) > 1500:  # Allow some flexibility
            chunk["content"] = " ".join(words[:1500]) + "..."
        
        # Add consistent global metadata using Python (no LLM variability)
        validated_chunk = {
            # Global metadata - consistent across all chunks
            "Insurer": global_metadata.get("insurer", "Unknown"),
            "Document_Name": global_metadata.get("document_name", "Unknown"),
            "Document_Date": global_metadata.get("document_date", "Unknown"),
            "Product_type": global_metadata.get("product_type", "Unknown"),
            "Page_no": page_num,
            
            # Page-level metadata from LLM - can vary by page/chunk
            "Section_Title": chunk["Section_Title"],
            "Subheading": chunk["Subheading"], 
            "content": chunk["content"],
            "content_label": chunk.get("content_label", "text")
        }
        
        return validated_chunk

    def _create_failure_chunk(self, page_num: int, global_metadata: Dict[str, str], 
                             error_msg: str, failure_type: str = "error", 
                             previous_context: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Create chunk when processing fails - unified method for both error and fallback cases"""
        
        # For fallback type, try to inherit context; for error type, use fixed values
        if failure_type == "fallback" and previous_context:
            section_title = previous_context.get("Section_Title", "N/A")
            subheading = previous_context.get("Subheading", "N/A")
        else:
            section_title = "Error"
            subheading = "Processing Error"
        
        return [{
            "Insurer": global_metadata.get("insurer", "Unknown"),
            "Document_Name": global_metadata.get("document_name", "Unknown"),
            "Document_Date": global_metadata.get("document_date", "Unknown"),
            "Section_Title": section_title,
            "Subheading": subheading,
            "Product_type": global_metadata.get("product_type", "Unknown"),
            "Page_no": page_num,
            "content": error_msg[:2000],  # Limit length consistently
            "content_label": failure_type
        }]



    def process_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> str:
        """
        Process entire PDF and return path to output JSONL file
        
        Args:
            pdf_path: Path to input PDF
            output_dir: Output directory (defaults to Data_Json)
            
        Returns:
            Path to generated JSONL file
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(pdf_path), "Data_Json")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        base_name = Path(pdf_path).stem
        output_file = os.path.join(output_dir, f"{base_name}_chunks.jsonl")
        
        print(f"Processing PDF: {pdf_path}")
        print(f"Output file: {output_file}")
        
        # Extract global metadata
        print("Extracting document metadata...")
        global_metadata = self.extract_global_metadata(pdf_path)
        print(f"Global metadata: {global_metadata}")
        
        # Convert PDF to images
        print("Converting PDF to images...")
        images = self.pdf_to_images(pdf_path)
        print(f"Converted {len(images)} pages")
        
        # Process each page
        all_chunks = []
        current_context = None  # Initialize context
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for img, page_num in images:
                print(f"Processing page {page_num}/{len(images)}...")
                chunks, current_context = self.process_page(img, page_num, global_metadata, current_context)
                
                # Write chunks to JSONL file
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                    all_chunks.append(chunk)
                
                print(f"  Generated {len(chunks)} chunks")
                if current_context:
                    print(f"  Context for next page: Section='{current_context.get('Section_Title', 'N/A')}', Subheading='{current_context.get('Subheading', 'N/A')}'")
        
        print(f"\nProcessing complete!")
        print(f"Total chunks generated: {len(all_chunks)}")
        print(f"Output saved to: {output_file}")
        
        # Generate summary statistics
        self._print_processing_summary(all_chunks, output_file)
        
        return output_file

    def _print_processing_summary(self, chunks: List[Dict[str, Any]], output_file: str):
        """Print processing summary statistics"""
        if not chunks:
            return
        
        print(f"\n{'='*50}")
        print("PROCESSING SUMMARY")
        print(f"{'='*50}")
        
        print(f"Total chunks: {len(chunks)}")
        print(f"Output file: {output_file}")
        
        # Content length statistics
        content_lengths = [len(chunk['content'].split()) for chunk in chunks]
        print(f"\nChunk size statistics (words):")
        print(f"  Average: {sum(content_lengths) / len(content_lengths):.1f}")
        print(f"  Min: {min(content_lengths)}")
        print(f"  Max: {max(content_lengths)}")
        
        # Pages covered
        pages = set(chunk['Page_no'] for chunk in chunks)
        print(f"\nPages processed: {min(pages)} - {max(pages)} ({len(pages)} total)")
        
        # Section distribution
        sections = [chunk['Section_Title'] for chunk in chunks if chunk['Section_Title'] != 'N/A']
        section_counts = {}
        for section in sections:
            section_counts[section] = section_counts.get(section, 0) + 1
        
        if section_counts:
            print(f"\nTop sections:")
            for section, count in sorted(section_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {section}: {count} chunks")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDS PDF using Gemini OCR")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output-dir", help="Output directory (default: Data_Json)")
    parser.add_argument("--api-key", help="Google API key (or set GOOGLE_API_KEY env var)")
    
    args = parser.parse_args()
    
    try:
        processor = GeminiPDSProcessor(api_key=args.api_key)
        output_file = processor.process_pdf(args.pdf_path, args.output_dir)
        print(f"\nSuccess! Output saved to: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
