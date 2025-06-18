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
import google.generativeai as genai
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
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # System prompt for document understanding and chunking
        self.system_prompt = """
You are an expert document processing AI specializing in insurance Product Disclosure Statements (PDS). 
Your task is to analyze each page image and extract structured content with precise metadata.

CRITICAL REQUIREMENTS:
1. Extract coherent chunks based on document structure (headings, sections, paragraphs)
2. Each chunk should be 50-1000 words and represent a logical unit of content
3. Preserve hierarchical relationships (sections, subsections)
4. Extract ALL required metadata fields for each chunk
5. Use heading detection to determine section structure
6. Handle tables, lists, and complex layouts intelligently

For each page, return a JSON array of chunks following this EXACT schema:
{
  "Insurer": "string",
  "Document_Name": "string", 
  "Document_Date": "YYYY-MM-DD",
  "Section_Title": "string",
  "Subheading": "string",
  "Captions": "string",
  "Product_type": "string",
  "Page_no": number,
  "content": "string"
}

METADATA EXTRACTION RULES:
- Insurer: Extract from header, footer, or document title
- Document_Name: Use document title or filename
- Document_Date: Find publication/effective date in ISO format
- Section_Title: Top-level section (e.g., "Section 2: Benefits")
- Subheading: Immediate subsection heading
- Captions: Any table/figure captions associated with the chunk
- Product_type: Infer from content (Life, TPD, Income Protection, etc.)
- Page_no: Current page number
- content: Main body text of the chunk

If any field cannot be determined, use "N/A" or "Unknown".
Ensure each chunk is meaningful and self-contained while respecting document structure.
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
            for page_num in range(min(3, len(doc))):  # First 3 pages
                page = doc.load_page(page_num)
                mat = fitz.Matrix(2, 2)  # Higher resolution for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                first_pages.append(img)
            
            # Use Gemini to extract metadata from first pages
            if first_pages:
                global_metadata = self._extract_document_metadata(first_pages)
                metadata.update(global_metadata)
                
        except Exception as e:
            print(f"Warning: Could not extract global metadata: {e}")
        
        doc.close()
        return metadata

    def _extract_document_metadata(self, images: List[Image.Image]) -> Dict[str, str]:
        """Use Gemini to extract document-level metadata from first few pages"""
        prompt = """
Analyze these first few pages of an insurance document and extract:
1. Insurer name (company providing the insurance)
2. Document name/title
3. Document date (publication or effective date in YYYY-MM-DD format)
4. Primary product type (Life, TPD, Trauma, Income Protection, etc.)

Return only a JSON object with these keys: insurer, document_name, document_date, product_type
If any information is not found, use "Unknown".
"""
        
        try:
            # Prepare content for Gemini
            content = [prompt]
            for img in images[:2]:  # Limit to first 2 pages for metadata
                content.append(img)
            
            response = self.model.generate_content(content)
            
            # Parse JSON response
            json_match = re.search(r'\{[^}]+\}', response.text)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"Warning: Could not extract metadata with Gemini: {e}")
        
        return {}

    def process_page(self, image: Image.Image, page_num: int, global_metadata: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process a single page and extract chunks"""
        try:
            content = [
                self.system_prompt,
                f"\nProcess this page {page_num} of an insurance PDS document.",
                f"Global document metadata: {json.dumps(global_metadata)}",
                "\nExtract chunks following the specified JSON schema:",
                image
            ]
            
            response = self.model.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistency
                    max_output_tokens=8192
                )
            )
            
            # Extract JSON from response
            chunks = self._parse_chunks_response(response.text, page_num, global_metadata)
            return chunks
            
        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
            return self._create_error_chunk(page_num, global_metadata, str(e))

    def _parse_chunks_response(self, response_text: str, page_num: int, global_metadata: Dict[str, str]) -> List[Dict[str, Any]]:
        """Parse Gemini response and extract chunks"""
        chunks = []
        
        # Try to find JSON arrays in the response
        json_pattern = r'\[.*?\]'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        if not json_matches:
            # Try to find individual JSON objects
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
                            # Validate and clean chunk
                            cleaned_chunk = self._validate_chunk(chunk, page_num, global_metadata)
                            if cleaned_chunk:
                                chunks.append(cleaned_chunk)
                break  # Use first valid JSON array
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, create fallback chunk
        if not chunks:
            chunks = self._create_fallback_chunk(response_text, page_num, global_metadata)
        
        return chunks

    def _validate_chunk(self, chunk: Dict[str, Any], page_num: int, global_metadata: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Validate and standardize chunk format"""
        required_fields = [
            "Insurer", "Document_Name", "Document_Date", "Section_Title", 
            "Subheading", "Captions", "Product_type", "Page_no", "content"
        ]
        
        # Ensure all required fields exist
        for field in required_fields:
            if field not in chunk:
                chunk[field] = "N/A"
        
        # Override with global metadata where appropriate
        chunk["Page_no"] = page_num
        if global_metadata.get("insurer") and global_metadata["insurer"] != "Unknown":
            chunk["Insurer"] = global_metadata["insurer"]
        if global_metadata.get("document_name") and global_metadata["document_name"] != "Unknown":
            chunk["Document_Name"] = global_metadata["document_name"]
        if global_metadata.get("document_date") and global_metadata["document_date"] != "Unknown":
            chunk["Document_Date"] = global_metadata["document_date"]
        if global_metadata.get("product_type") and global_metadata["product_type"] != "Unknown":
            chunk["Product_type"] = global_metadata["product_type"]
        
        # Validate content length (50-1000 words)
        content = str(chunk.get("content", "")).strip()
        if not content or len(content.split()) < 5:  # Too short
            return None
        
        # Truncate if too long (keep within reasonable limits)
        words = content.split()
        if len(words) > 1500:  # Allow some flexibility
            chunk["content"] = " ".join(words[:1500]) + "..."
        
        return chunk

    def _create_error_chunk(self, page_num: int, global_metadata: Dict[str, str], error_msg: str) -> List[Dict[str, Any]]:
        """Create error chunk when processing fails"""
        return [{
            "Insurer": global_metadata.get("insurer", "Unknown"),
            "Document_Name": global_metadata.get("document_name", "Unknown"),
            "Document_Date": global_metadata.get("document_date", "Unknown"),
            "Section_Title": "Error",
            "Subheading": "Processing Error",
            "Captions": "N/A",
            "Product_type": global_metadata.get("product_type", "Unknown"),
            "Page_no": page_num,
            "content": f"Error processing page {page_num}: {error_msg}"
        }]

    def _create_fallback_chunk(self, response_text: str, page_num: int, global_metadata: Dict[str, str]) -> List[Dict[str, Any]]:
        """Create fallback chunk from raw response when JSON parsing fails"""
        # Clean up the response text
        content = re.sub(r'[^\w\s.,;:!?()-]', ' ', response_text)
        content = re.sub(r'\s+', ' ', content).strip()
        
        if len(content) < 50:  # Too short to be useful
            content = f"Unable to process page {page_num} content properly."
        
        return [{
            "Insurer": global_metadata.get("insurer", "Unknown"),
            "Document_Name": global_metadata.get("document_name", "Unknown"), 
            "Document_Date": global_metadata.get("document_date", "Unknown"),
            "Section_Title": "N/A",
            "Subheading": "N/A",
            "Captions": "N/A",
            "Product_type": global_metadata.get("product_type", "Unknown"),
            "Page_no": page_num,
            "content": content[:2000]  # Limit length
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
        with open(output_file, 'w', encoding='utf-8') as f:
            for img, page_num in images:
                print(f"Processing page {page_num}/{len(images)}...")
                chunks = self.process_page(img, page_num, global_metadata)
                
                # Write chunks to JSONL file
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                    all_chunks.append(chunk)
                
                print(f"  Generated {len(chunks)} chunks")
        
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
