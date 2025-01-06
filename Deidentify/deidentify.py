import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)


'''from transformers import pipeline
Pipeline = pipeline(
    "text-generation",
    model=model_name,
    device_map = "auto",
    tokenizer = tokenizer
)'''

'''#model = AutoModelForCausalLM.from_pretrained(model_name)
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_name)
model = load_checkpoint_and_dispatch(model, model_name, device_map="auto")

# Move model to multiple GPUs
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to("cuda")
'''
# Define the directory containing clinical notes
notes_directory = "Notes"  # Replace with your directory path
output_directory = "deidNotes"   # Replace with your desired output path

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Define the prompt for de-identification
deidentification_prompt = "De-identify the following clinical note:\n"

# Function to de-identify a single clinical note
def deidentify_clinical_notes(note_text):
        # Construct the input text with the prompt
        input_text = deidentification_prompt + note_text
        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda")
        # Generate de-identified text
        with torch.no_grad():
                outputs = model.generate(
                inputs["input_ids"],
                max_length=512,  # Adjust as needed
                num_beams=5,     # Beam search for better results
                early_stopping=True
                )
        # Decode the generated text
        deidentified_note = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return deidentified_note

# Process each clinical note in the folder
for filename in os.listdir(notes_directory):
    if filename.endswith(".txt"):  # Assuming clinical notes are text files
        file_path = os.path.join(notes_directory, filename)

        with open(file_path, "r") as file:
            note_text = file.read()

        # De-identify the clinical note
        deidentified_note = deidentify_clinical_notes(note_text)

        # Save the de-identified note to
        output_path = os.path.join(output_directory, filename)
        with open(output_path, "w") as output_file:
            output_file.write(deidentified_note)

        print(f"Processed {filename}")

print("All clinical notes have been de-identified and saved.")
