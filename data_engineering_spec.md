# Data Engineering Pipeline Specification: PDF to SFT/DPO Dataset

**Objective:** Define the programmatic framework for extracting, enriching, and synthesizing raw PDF documents into a high-fidelity dataset optimized for 1B parameter language model alignment.

---

## 1. Ingestion & Layout-Aware Parsing

PDFs strip away semantic structure in favor of visual coordinates. The pipeline must reconstruct this logic before text generation occurs.

* **Layout Parsing Tooling:** The script must utilize layout-aware libraries such as `marker-pdf` or `unstructured.io` instead of raw OCR (like Tesseract or PyPDF2).
* **Table Reconstruction:** Multi-column troubleshooting tables must be programmatically flattened into linear Markdown (e.g., `Problem: [X] | Solution: [Y]`) to preserve the causal relationship.
* **Boilerplate Filtering:** Implement regex or heuristic filters to aggressively drop repeating page numbers, FCC compliance warnings, copyright footers, and non-actionable warranty information.

---

## 2. Contextual Chunking & Metadata Injection

To prevent the "Amnesia Problem" where the fine-tuned model provides generic or incorrect advice for a specific printer, context must be forcefully attached to every piece of data.

* **Semantic Boundaries:** Text must be chunked based on Markdown headers (e.g., H2 or H3) rather than arbitrary character counts, keeping troubleshooting steps organically grouped.
* **Token Limits:** Restrict chunks to a maximum of 1,000 tokens to prevent the Teacher Model's attention mechanism from degrading during synthesis.
* **The Metadata Header:** Before a chunk is sent to the Teacher Model, the script MUST prepend a standardized context header. 
* **Header Example:** `[Source Context: Product User Guide] [Section: Troubleshooting]`

---

## 3. SFT Synthesis: The Diversity Matrix

The pipeline must prioritize the "Less is More for Alignment" (LIMA) principle. 3,000 highly diverse, impeccably formatted examples will outperform 20,000 repetitive ones.

* **Synthesis Quota:** Instruct the Teacher Model to generate exactly 3 QA pairs per valid chunk.
* **The Multi-Angle Prompting Rule:** The Teacher Model must frame the 3 questions from distinct user perspectives:
    * **Symptom-Based:** "My device has a blinking orange light and won't connect to my network."
    * **Direct Action:** "How do I factory reset the network settings on this model?"
    * **Clarification/Edge-Case:** "Does the reset button need to be held down, or just pressed once?"
* **Response Formatting:** Enforce that the "chosen" response always mentions the specific source document context within its answer to reinforce the model's contextual grounding.

---

## 4. DPO Synthesis: The Hard Negative Protocol

Direct Preference Optimization requires the model to learn the boundary between helpfulness and plausible hallucination. The "rejected" examples must be technically dangerous, not linguistically poor.

* **Plausibility Constraint:** The rejected answer must mirror the tone, formatting, and confidence of the chosen answer. 
* **The Flaw Injection Matrix:** Instruct the Teacher Model to inject one of the following specific flaw types into the rejected response:
    * **Wrong Sequence:** Instructing the user to execute step 4 before step 1 (e.g., performing an action before the prerequisite step).
    * **Wrong Component:** Diagnosing one symptom as an unrelated issue (e.g., confusing a hardware problem for a software configuration issue).
    * **Subtle Hallucination:** Inventing a nonexistent menu path (e.g., "Navigate to Settings -> Advanced -> Clear Cache").

---

## 5. Quality Control & Deduplication Pipeline

An automated script processing 50 manuals will inevitably generate overlapping questions (e.g., 50 different ways to ask "How to load paper"). 

* **Embedding Extraction:** Pass all generated "prompts" through a lightweight embedding model (such as `all-MiniLM-L6-v2` via Hugging Face `sentence-transformers`).
* **Cosine Similarity Threshold:** Calculate the similarity score between new prompts and the existing dataset.
* **Pruning Logic:** If a new prompt has a cosine similarity $> 0.85$ with any existing prompt, the script must discard the new QA pair. This forces the dataset to remain strictly diverse and prevents the 1B model from overfitting on common queries.
* **Final Validation:** Assert that all retained JSON objects contain non-null `prompt`, `chosen`, and `rejected` keys before appending to the final `hp_alignment_dataset.jsonl` file.
