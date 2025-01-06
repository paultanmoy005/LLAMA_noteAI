import csv
import time
import torch
import os
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb
import random
from accelerate.utils import gather_object
import sys

def reset_env():
    torch.cuda.empty_cache()
    os.system("reset_system_command")
    random.seed(42)
    torch.manual_seed(42)

def load_clinical_notes(directory):
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt")]
    notes = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            notes.append((file_path, file.read().strip()))
    return notes

def write_summary(file_path, summary):
    summary_file_path = file_path.replace(".txt", "_summary.txt")
    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write(summary)

def setupModelTokenizer(modelName, bit):
    #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))
    if bit==4:
        model = AutoModelForCausalLM.from_pretrained(modelName, load_in_4bit=True, device_map="auto")
    elif bit==8:
        model = AutoModelForCausalLM.from_pretrained(modelName, load_in_8bit=True, device_map="auto")
    elif bit==16:
        model = AutoModelForCausalLM.from_pretrained(modelName, torch_dtype=torch.float16, device_map="auto")
    elif bit==32:
        model = AutoModelForCausalLM.from_pretrained(modelName, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def prepare_notes(notes, tokenizer, batch_size):
    prompt_template = '''<s>[INST] <<SYS>> Identify a patient's sexual orientation from the text. Your response should be in this format: "Sexual Orientation" > (extracted orientation from \
    text). If no information about the sexual orientation is provided in the text, return "Sexual Orientation" > "Not Specified" <</SYS>>
    {}
    [/INST]'''
    batches = [notes[i:i + batch_size] for i in range(0, len(notes), batch_size)]
    batches_tok = []
    tokenizer.padding_side = "left"
    for note_batch in batches:
        file_paths, note_texts = zip(*note_batch)
        prompts = [prompt_template.format(note) for note in note_texts]
        tokenized_batch = tokenizer(
            prompts,
            return_tensors="pt",
            padding='longest',
            truncation=True,
            add_special_tokens=True
        ).to("cuda")
        batches_tok.append((file_paths, tokenized_batch))
    tokenizer.padding_side = "right"
    return batches_tok

def run_inference(batch_size, inDir, modelName, bit):
    model, tokenizer = setupModelTokenizer(modelName,bit)
    notes_all = load_clinical_notes(inDir)

    # Synchrnoize GPUs and start the timer
    accelerator.wait_for_everyone()
    start = time.time()

    #Divide the notes list among available GPUs
    with accelerator.split_between_processes(notes_all) as notes:
        results = []

        # Prepare notes for tokenization
        note_batches = prepare_notes(notes, tokenizer, batch_size)

        # Summarize each note batch
        for file_paths, tokenized_batch in note_batches:
            summaries_tokenized = model.generate(
                **tokenized_batch,
                max_new_tokens=20,  # Adjust based on model's token output length
            )

            # Decode and truncate to 200 words
            summaries = tokenizer.batch_decode(summaries_tokenized, skip_special_tokens=True)
            truncated_summaries = [' '.join(summary.split()[:20]) for summary in summaries]

            # Save summaries
            #for file_path, summary in zip(file_paths, truncated_summaries):
                #write_summary(file_path, summary)
                #results.append((file_path, summary))'''

        # Transform results to list for gather_object()
        results = [results]

    # Collect results from all GPUs
    results_gathered = gather_object(results)
    if accelerator.is_main_process:
        timediff = time.time() - start
    return timediff

# Main script to iterate over batch sizes and number of GPUs
#batch_sizes = [2,4]  # Batch sizes from 1 to 10
#num_gpus_list = [2,4]  # Number of GPUs from 2 to 8
#bits=[4,8,16,32]
if __name__=="main":
    bit = sys.argv[1]
    batch_size = sys.argv[2]
    num_gpus = sys.argv[3]
    sourceDir = sys.argv[4]
    destDir = sys.argv[5]
    accelerator = Accelerator()
    with open(destDir, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Bit ", "Batch Size ", "Number of GPUs ", "Inference Time (seconds)"])
        for batch_size in batch_sizes:
            for bit in bits:
                reset_env()
                #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
                #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))
                num_gpus = torch.cuda.device_count()
                print(f"Number of available GPUs: {num_gpus}")
                try:
                    inference_time = run_inference(batch_size, sourceDir, "Meta-Llama/Meta-Llama-3-8B-Instruct", bit)
                    writer.writerows([[bit, batch_size, num_gpus, inference_time]])
                    print(f"Bit size:{bit}, Batch size: {batch_size}, GPUs: {num_gpus}, Time: {inference_time}")
                except Exception as e:
                    print(f"An error occured with {bit} bit, {batch_size} batch, {num_gpus} gpus: {e}")