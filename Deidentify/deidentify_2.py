import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# Load the model and tokenizer
model_name = "Meta-Llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
# Define the directory containing clinical notes
notes_directory = "syntheticNotes" # Replace with your directory
output_directory = "deidNotes" # Replace with your desired output path
# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)
# Define the prompt for de-identification
deidentification_prompt = "De-identify the following clinical note:\n"
# Function to de-identify a single clinical note
tokenizer.pad_token = tokenizer.eos_token
def deidentify_clinical_notes(note_text):
        # Construct the input text with the prompt
        input_text = deidentification_prompt + note_text
        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        # Generate de-identified text
        with torch.no_grad():
                outputs = model.generate( inputs["input_ids"], max_new_tokens=500)
        # Decode the generated text
        deidentified_note = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return deidentified_note
# Process each clinical note in the folder
for filename in os.listdir(notes_directory):
        if filename.endswith(".txt"): # Assuming clinical notes are text file
                file_path = os.path.join(notes_directory, filename)
                with open(file_path, "r") as file:
                        note_text = file.read()
                # De-identify the clinical note
                deidentified_note = deidentify_clinical_notes(note_text)
                # Save the de-identified note to the output directory
                output_path = os.path.join(output_directory, filename)
                with open(output_path, "w") as output_file:
                        output_file.write(deidentified_note)
print(f"Processed {filename}")
print("All clinical notes have been de-identified and saved.")