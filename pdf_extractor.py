import os
from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker

class DoclingManualExtractor:
    def __init__(self) -> None:
        """
        Initializes the Docling converter and its native hierarchical chunker.
        Docling uses advanced ML models (DocLayNet) under the hood to understand 
        the visual structure of the PDF before extracting text.
        """
        print("Initializing Docling Document Converter...")
        self.converter = DocumentConverter()
        self.chunker = HierarchicalChunker()

    def process_manual(self, pdf_path: str, device_context: str) -> list[str]:
        """
        Parses a PDF manual, perfectly preserves tables/columns as Markdown,
        chunks the document based on its internal heading structure, and injects metadata.
        """
        print(f"\nAnalyzing Structural Layout from: {pdf_path}...")
        
        # 1. High-Fidelity Conversion
        # This step uses vision models to identify tables, headers, and reading order.
        conv_result = self.converter.convert(pdf_path)
        
        # 2. Semantic Chunking
        # Docling natively understands the document tree (Heading 1 -> Heading 2 -> Paragraph)
        # and chunks the text logically based on that hierarchy.
        chunks = self.chunker.chunk(conv_result.document)

        # 3. Metadata Injection & Filtering
        final_dataset_chunks = []
        for chunk in chunks:
            # Extract the clean Markdown text from the chunk
            raw_text = chunk.text
            
            # Filter out tiny, useless chunks (like isolated page numbers or logos)
            if len(raw_text.strip()) < 100:
                continue 
            
            # Format as clean Markdown with the injected context header
            enriched_chunk = (
                f"### [Source Context: {device_context}]\n\n"
                f"{raw_text}\n"
            )
            final_dataset_chunks.append(enriched_chunk)

        print(f"Extraction complete. Yielded {len(final_dataset_chunks)} high-fidelity chunks.")
        return final_dataset_chunks

# ==============================================================================
# Execution Example for the AI Agent
# ==============================================================================
if __name__ == "__main__":
    extractor = DoclingManualExtractor()
    
    # Example Target
    target_pdf = "example_manual.pdf"
    source_name = "Product User Guide"
    
    if os.path.exists(target_pdf):
        enriched_chunks = extractor.process_manual(target_pdf, source_name)
        
        if enriched_chunks:
            print("\n--- PREVIEW OF DOCLING CHUNK 1 ---")
            print(enriched_chunks[0])
            print("----------------------------------")
    else:
        print(f"Awaiting PDF file: {target_pdf}")
    