from huggingface_hub import InferenceClient  # importing the InferenceClient from Hugging Face Hub
# the Llama3 model is too large to load automatically, so had to use the Hugging Face Inference API to access it.
import os 
from dotenv import load_dotenv  


load_dotenv() 

def get_model():
    # creating a model client
    model = InferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct", token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"))  
    return model  

def run_inference(prompt: str):
    # obtaining the model client
    client = get_model()  

    output = client.text_generation(prompt=prompt, max_new_tokens=2048, temperature=0.1)  # model will generate text based on the input prompt
    return output  

def chat_with_llama():
    print("Chat with Llama-3! Type 'quit' to exit.")  
    while True:  # infinite loop starts ,  this line creates an infinite loop. The loop will continue indefinitely because True is always true.
        # Get input from the user
        user_input = input("You: ")  
        if user_input.lower() == 'quit':  #  if the user types "quit", the break statement is executed, which immediately terminates the loop
            break  
        # getting the model response
        response = run_inference(user_input)  
        print("Llama-3:", response)  

if __name__ == "__main__":
    chat_with_llama()  
