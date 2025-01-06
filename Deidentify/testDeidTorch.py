import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast

# Custom dataset to handle the clinical notes
class ClinicalNotesDataset(Dataset):
    def __init__(self, directory, tokenizer, prompt):
        self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt")]
        self.tokenizer = tokenizer
        self.prompt = prompt

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, 'r') as file:
            note_text = file.read()
        input_text = self.prompt + note_text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        return inputs["input_ids"].squeeze(0), file_path


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # or the IP address of the master node
    os.environ['MASTER_PORT'] = '29500'      # a free port on the master node
    os.environ['WORLD_SIZE'] = str(world_size)  # the total number of processes
    os.environ['RANK'] = str(rank)  # the rank of this process
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def deidentify_notes(rank, world_size, notes_directory, output_directory, model_name):
    setup(rank, world_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    prompt = "De-identify the following clinical note:\n"
    dataset = ClinicalNotesDataset(notes_directory, tokenizer, prompt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    os.makedirs(output_directory, exist_ok=True)

    for input_ids, file_path in dataloader:
        input_ids = input_ids.to(rank)
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=500)
        deidentified_note = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Save the de-identified note to the output directory
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_directory, filename)
        with open(output_path, "w") as output_file:
            output_file.write(deidentified_note)
        print(f"Processed {filename} on GPU {rank}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    notes_directory = "syntheticNotes"  # Replace with your directory
    output_directory = "deidNotes"  # Replace with your desired output path
    model_name = "Meta-Llama/Meta-Llama-3-8B-Instruct"

    mp.spawn(
        deidentify_notes,
        args=(world_size, notes_directory, output_directory, model_name),
        nprocs=world_size,
        join=True
    )
    print("All clinical notes have been de-identified and saved.")