import json
import torch
from sentence_transformers import SentenceTransformer, util

class DatasetFilter:
    def __init__(self, similarity_threshold: float = 0.85) -> None:
        """
        Initializes the embedding model and the vector store.
        Using all-MiniLM-L6-v2 as it is extremely fast and perfect for sentence-level semantic matching.
        """
        print("Loading Embedding Model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = similarity_threshold
        
        self.accepted_data = []
        self.accepted_embeddings = None

    def validate_schema(self, qa_pair: dict) -> bool:
        """Ensures the generated tuple contains exactly what the SFT/DPO trainers expect."""
        required_keys = ["prompt", "chosen", "rejected"]
        
        # Check for missing keys
        if not all(key in qa_pair for key in required_keys):
            return False
            
        # Check for empty or null strings
        if not all(isinstance(qa_pair[key], str) and len(qa_pair[key].strip()) > 0 for key in required_keys):
            return False
            
        return True

    def process_new_pair(self, qa_pair: dict) -> bool:
        """
        Validates the schema, calculates semantic similarity against all accepted prompts,
        and decides whether to keep or discard the new pair.
        """
        if not self.validate_schema(qa_pair):
            print("Dropped: Failed schema validation (missing keys or empty strings).")
            return False

        new_prompt = qa_pair["prompt"]
        new_embedding = self.model.encode(new_prompt, convert_to_tensor=True)

        # If this is the first item, accept it automatically
        if self.accepted_embeddings is None:
            self.accepted_embeddings = new_embedding.unsqueeze(0)
            self.accepted_data.append(qa_pair)
            return True

        # Calculate cosine similarity against all previously accepted prompts
        cosine_scores = util.cos_sim(new_embedding, self.accepted_embeddings)[0]
        max_similarity = torch.max(cosine_scores).item()

        if max_similarity > self.similarity_threshold:
            print(f"Dropped: Semantic similarity too high ({max_similarity:.2f}) -> '{new_prompt}'")
            return False
        else:
            # Append new embedding to the vector store
            self.accepted_embeddings = torch.cat((self.accepted_embeddings, new_embedding.unsqueeze(0)), dim=0)
            self.accepted_data.append(qa_pair)
            return True

    def save_dataset(self, output_path: str) -> None:
        """Saves the strictly filtered dataset to a JSONL file for training."""
        print(f"\nSaving {len(self.accepted_data)} highly diverse QA pairs to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.accepted_data:
                f.write(json.dumps(item) + '\n')
        print("Dataset successfully compiled.")

# ==============================================================================
# Execution Example for the AI Agent
# ==============================================================================
if __name__ == "__main__":
    # Initialize the filter with the strict 0.85 threshold
    qa_filter = DatasetFilter(similarity_threshold=0.85)
    
    # Example simulated outputs from the Teacher Model
    synthetic_outputs = [
        {
            "prompt": "How do I clear a paper jam in the automatic document feeder?",
            "chosen": "Lift the cover and gently pull the paper...",
            "rejected": "Yank the paper out forcefully..."
        },
        {
            "prompt": "What is the best way to fix a document feeder jam?",
            "chosen": "Open the top hatch and roll the paper out...",
            "rejected": "Turn the device upside down and shake it..."
        },
        {
            "prompt": "Why is my device not connecting to the 5GHz Wi-Fi network?",
            "chosen": "Ensure your router is broadcasting a 2.4GHz band, as many older devices do not support 5GHz...",
            "rejected": "Devices do not use Wi-Fi, you must use a USB cable."
        },
        {
            "prompt": "", # Will trigger schema failure
            "chosen": "Some answer",
            "rejected": "Some wrong answer"
        }
    ]

    print("--- Starting Dataset Deduplication Pipeline ---")
    for pair in synthetic_outputs:
        qa_filter.process_new_pair(pair)
        
    qa_filter.save_dataset("alignment_dataset.jsonl")
