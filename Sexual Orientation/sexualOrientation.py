model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

from transformers import pipeline
Pipeline = pipeline(
    "text-generation",
    model=model_name,
    device_map = "auto",
    tokenizer = tokenizer
)

# Define the prompt
prompt_template = '''<s>[INST] <<SYS>>
Identify a patient's sexual orientation from the text. Your response should be in this format: "Sexual Orientation" >
(extracted orientation from text). If no information about the sexual orientation is provided in the text, return "S>
Orientation" > "Not Specified" <</SYS>>
{}
[/INST]'''

# List of text file paths
path = "/home/ssm-user/NoteAI/syntheticNotes/"
text_files=[]
for i in range(1,11):
  text_files.append(f"note{i}.txt")

# Iterate through each file, read its content, and summarize it
summaries = {}
for file_path in text_files:
    with open(path+file_path, 'r') as file:
        content = file.read()

    # Create the prompt with the content of the file
    prompt = prompt_template.format(content)

    # Generate summary
    summary = Pipeline(prompt, max_new_tokens=20, return_full_text=False, num_return_sequences=1)[0]['generated_text>
    summaries[file_path] = summary

for file_path, summary in summaries.items():
    print(f"Summary of {file_path}:\n{summary}\n")