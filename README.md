# NotesAI
The purpose of this repository is to document the work of clinical notes processing using the Artificial Intelligence and Large Language Modeling methods and technologies
This instruction file describes each folder:

# Docker Setup
It contains the Dockerfile to build a docker container with amazon linux image and create the environemnt for instaling required python libraries for LLM studies. It also comes with a python script "downloadLLM.py" to download any huggingface model using the installed libraries

# Chat
testPipeline.py > it uses the transformer pipeline function from huggingface for continous chat using an LLM

# Fine-tune
fineTune_llama2_chat.py > demonstration of fine-tunig llama-2-7b-chat with huggingface dataset "mlabonne/guanaco-llama2-1k". It shows how fine-tuning can be achieved using a singe GPU by applying LoRA technique

fineTune_llama2_chat_icd10.py > demonstration of fine-tuning using ICD-10 dataset.

# Deidentify
deidentify and deidentify_2 demonstrate the deidentification task of llm using different set of parameters. testDeidTorch.py is a template for deidentification of notes in multiple GPU settings using pyTorch.

# Sexual Orientation
testCuda.py > number of GPUs can be defined from the terminal using the following command:
export CUDA_VISIBLE_DEVICES=0,1,2,3
This command will expose GPU 0 to 4. The python file is a python alternative of the command.
sexualOrientation.py > demonstrates how the prompt should be desinged for extracting sexual orinetation related infromation from the notes.
sexualOrientation_GPU_configuration.py > Upon receiving the parameter from the terminal, this script will extract sexual orientation infromation form the text using different values of bit size, batch size and GPU no. As output, it will generate a .csv file with operationt time for each value set of bit size, batch size and GPU no.
