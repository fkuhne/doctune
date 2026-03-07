import os
import glob
import time

# Import the three pillars we built previously
# (Ensure pdf_extractor.py, teacher_model_synthesis.py, and deduplicate_dataset.py are in the same folder)
from pdf_extractor import DoclingManualExtractor
from teacher_model_synthesis import TeacherModelSynthesizer
from deduplicate_dataset import DatasetFilter

class DatasetBuilder:
    def __init__(self, input_dir: str = "./manuals", output_file: str = "alignment_dataset.jsonl") -> None:
        """
        Initializes the master orchestrator.
        """
        self.input_dir = input_dir
        self.output_file = output_file
        
        print("--- INITIALIZING PHASE 2 PIPELINE ---")
        self.extractor = DoclingManualExtractor()
        self.synthesizer = TeacherModelSynthesizer() # Requires OPENAI_API_KEY in env
        self.filter = DatasetFilter(similarity_threshold=0.85)

    def extract_device_context(self, filename: str) -> str:
        """
        A helper function to convert a filename like 'product_user_guide.pdf'
        into a clean context string: 'Product User Guide'.
        """
        base_name = os.path.basename(filename).replace(".pdf", "")
        clean_name = base_name.replace("_", " ").replace("-", " ").title()
        # Strip common trailing words
        clean_name = clean_name.replace(" Manual", "").replace(" User Guide", "")
        return clean_name

    def build(self) -> None:
        """
        The main execution loop. Finds all PDFs, extracts chunks, synthesizes data, 
        filters out duplicates, and saves the final dataset.
        """
        # Find all PDF manuals in the input directory
        pdf_files = glob.glob(os.path.join(self.input_dir, "*.pdf"))
        
        if not pdf_files:
            print(f"CRITICAL: No PDFs found in directory '{self.input_dir}'.")
            return

        print(f"Found {len(pdf_files)} manuals to process. Starting pipeline...\n")
        
        total_chunks_processed = 0
        total_pairs_generated = 0

        # Loop 1: Iterate through every PDF
        for i, pdf_path in enumerate(pdf_files):
            device_context = self.extract_device_context(pdf_path)
            print(f"============================================================")
            print(f"Processing Manual {i+1}/{len(pdf_files)}: {device_context}")
            print(f"============================================================")
            
            try:
                # Step 1: Extract and Chunk (Docling)
                enriched_chunks = self.extractor.process_manual(pdf_path, device_context)
                
                # Loop 2: Iterate through every chunk in the manual
                for j, chunk in enumerate(enriched_chunks):
                    total_chunks_processed += 1
                    print(f"  -> Synthesizing chunk {j+1}/{len(enriched_chunks)}...", end=" ", flush=True)
                    
                    # Step 2: Generate SFT and DPO data (OpenAI)
                    generated_tuples = self.synthesizer.process_chunk(chunk)
                    
                    if not generated_tuples:
                        print("[Skipped: No actionable data]")
                        continue
                        
                    # Loop 3: Filter and Deduplicate (Sentence Transformers)
                    kept_count = 0
                    for qa_tuple in generated_tuples:
                        total_pairs_generated += 1
                        # process_new_pair returns True if kept, False if dropped
                        if self.filter.process_new_pair(qa_tuple):
                            kept_count += 1
                            
                    print(f"[Generated {len(generated_tuples)} | Kept {kept_count}]")
                    
                    # Optional: Sleep briefly to avoid hitting OpenAI rate limits
                    time.sleep(1) 
                    
            except Exception as e:
                print(f"\nCRITICAL ERROR processing {pdf_path}: {e}")
                print("Skipping to the next manual to preserve pipeline execution...")

        # Final Step: Save the dataset
        print("\n============================================================")
        print("PIPELINE COMPLETE.")
        print(f"Total PDF Manuals Processed: {len(pdf_files)}")
        print(f"Total Semantic Chunks Analyzed: {total_chunks_processed}")
        print(f"Total QA Pairs Generated: {total_pairs_generated}")
        print(f"Total Unique QA Pairs Retained: {len(self.filter.accepted_data)}")
        print("============================================================")
        
        self.filter.save_dataset(self.output_file)

# ==============================================================================
# Execution Example for the AI Agent
# ==============================================================================
if __name__ == "__main__":
    # Agent Directive: Ensure the ./manuals folder exists and contains the PDFs
    os.makedirs("./manuals", exist_ok=True)
    
    builder = DatasetBuilder(
        input_dir="./manuals",
        output_file="alignment_dataset.jsonl"
    )
    
    # Run the pipeline
    builder.build()
