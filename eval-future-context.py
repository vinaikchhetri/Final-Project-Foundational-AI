from vllm import LLM
from vllm.sampling_params import SamplingParams
import pandas as pd
import re
import json


prompt_template = """You are a cybercrime classification expert. Classify the following text into one of these categories:

1. Child Pornography
2. Computer-related forgery
3. Computer-related fraud
4. Copyright-related offences
5. Cyber laundering
6. Cyberwarfare
7. Data Interference
8. Identity theft
9. Illegal Access (hacking, cracking)
10. Illegal data acquisition (data espionage)
11. Illegal gambling and online games
12. Illegal Interception
13. Misuse of devices
14. Phishing
15. Pornographic Material
16. Racism and hate speech on the Internet
17. Religious Offences
18. Spam and related threats
19. System Interference
20. Terrorist use of the Internet
21. Trademark-related offences
-1. Not related to cybercrime (use this if the text doesn't fit any cybercrime category)

Respond with ONLY the category number (-1 to 21). No extra text, no explanations."""

print("starting")

sp = SamplingParams(seed=1, temperature=0, top_p=1, max_tokens=5)  # max_tokens = 5 since we only need a number
print(repr(sp))

# models
#llm = LLM("mistralai/Mixtral-8x22B-Instruct-v0.1", tensor_parallel_size=4)
#llm = LLM("mistralai/Mixtral-8x7B-Instruct-v0.1", tensor_parallel_size=4)
#llm = LLM("mistralai/Mistral-7B-Instruct-v0.1", tensor_parallel_size=4)
#llm = LLM("meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=4)
llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=4)


# Load the data
df = pd.read_csv('ac_num.csv')
df = df.iloc[0:351] # only 351 labeled
print(f"Loaded {len(df)} texts")

batch = []
batch_size = 1000
prev_index = 0
targets = []

# Process in batches
for i in range(len(df)):
    text = df["text"].iloc[i]

    # Get next context if available
    next_context = df["next_context"].iloc[i] if "next_context" in df.columns else None
    
    # Trim very long texts 
    if len(text) > 1000:
        text = text[:1000] + "..."
    
    # Prepare user message
    if next_context and pd.notna(next_context):
        user_message = f"Next context: {next_context}\n\nText to classify: {text}"
    else:
        user_message = f"Classify this text: {text}"
    
    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": user_message}
    ]
    
    batch.append(messages)
    
    # Process batches of 1000
    if (i + 1) % batch_size == 0 or i == len(df) - 1:
        print(f"Processing batch: {prev_index} to {i}")
        outputs = llm.chat(batch, sampling_params=sp)
        
        # Extract results
        for j, output in enumerate(outputs):
            result = output.outputs[0].text.strip()
            # Extract number from response
            numbers = re.findall(r'-?\d+', result)
            if numbers:
                category = int(numbers[0])
                # Validate category
                if category == -1 or (1 <= category <= 21):
                    targets.append(category)
                else:
                    print(f"Invalid category {category} at index {prev_index + j}")
                    targets.append(None)
            else:
                print(f"No valid number found in response at index {prev_index + j}: {result}")
                targets.append(None)
        
        prev_index = i + 1
        batch = []

# Add predictions to dataframe
df['Meta-Llama-3-8B-Instruct'] = targets

# Save results
output_filename = 'Meta-Llama-3-8B-Instruct.csv'
df.to_csv(output_filename, index=False)

print(f"\nDone! Results saved to {output_filename}")
