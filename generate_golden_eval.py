import os
import json
from openai import OpenAI

# Initialize OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Define System Prompt focusing on multi-step reasoning
SYSTEM_PROMPT = """You are an expert technical support engineer specializing in HP printers. Your objective is to create highly complex, edge-case troubleshooting scenarios for HP printers.
You must focus purely on multi-step reasoning questions that require deep diagnostic logic.

Output exactly 10 scenarios.
Return the output strictly in JSON format using a single key "scenarios", which contains an array of objects.
Each object MUST have the following keys:
- "prompt": A detailed user query describing a complex, multi-layered HP printer issue.
- "chosen": The correct, step-by-step diagnostic and resolution process.
- "rejected": A plausible but incorrect or factually flawed resolution that might mislead a user or cause further issues.
"""

def generate_scenarios(batch_size=10):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate exactly {batch_size} complex, edge-case HP printer troubleshooting scenarios focusing on multi-step reasoning. Output valid JSON."}
        ],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    
    content = response.choices[0].message.content
    data = json.loads(content)
    return data.get("scenarios", [])

def main():
    target_count = 100
    batch_size = 10
    total_generated = 0
    output_file = "golden_eval.jsonl"
    
    print(f"Generating {target_count} complex HP printer scenarios...")
    
    scenarios = []
    # Loop until we reach target_count
    while total_generated < target_count:
        try:
            print(f"Generating batch... ({total_generated}/{target_count})")
            batch = generate_scenarios(batch_size)
            if not batch:
                continue
            scenarios.extend(batch)
            total_generated = len(scenarios)
        except Exception as e:
            print(f"Error generating batch: {e}")
            
    # Trim to exactly target_count
    scenarios = scenarios[:target_count]
    
    # Save to JSONL
    print(f"Saving to {output_file}...")
    with open(output_file, "w") as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario) + "\n")
            
    print(f"Successfully generated and saved {len(scenarios)} scenarios to {output_file}.")

if __name__ == "__main__":
    main()
