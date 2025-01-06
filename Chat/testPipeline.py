# import required libraries
from transformers import AutoTokenizer
import transformers
import torch

#specify the model and tokenizer
model = "meta-llama/Meta-Llama-3-8B" # meta-llama/Llama-2-7b-hf
tokenizer = AutoTokenizer.from_pretrained(model, token=True)

#build the transformer pipeline
from transformers import pipeline

llama_pipeline = pipeline(
    "text-generation",  # LLM task
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

#create a function for passing additional parameters and chat
def get_llama_response(prompt: str) -> None:
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response.
    """
    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1024,
    )
    print("Chatbot:", sequences[0]['generated_text'])


    #create a loop for continuious chatting
    while True:
    user_input = input("You: ")
    if user_input.lower() in ["bye", "quit", "exit"]:
        print("Chatbot: Goodbye!")
        break
    get_llama_response(user_input)