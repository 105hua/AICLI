"""
This is the package for the Local Text Generation Menu.
"""

# pylint: disable=C0301,E0401

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from modules import general

def menu():
    """
    This is the menu for Local Text Generation.
    :return:
    """
    model_id = None
    while model_id is None or model_id == "":
        general.clear_screen()
        print("Please enter the ID of the Model you'd like to use:")
        model_id = str(input())
    general.clear_screen()
    try:
        if "HF_API_KEY" in os.environ:
            bab_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=bab_config,
                token=os.environ["HF_API_KEY"]
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                padding_side="left",
                token=os.environ["HF_API_KEY"]
            )
        else:
            general.clear_screen()
            print("To use this feature, please set the HF_API_KEY in your .envrc file.")
            return
    except OSError:
        general.clear_screen()
        print("The Model could not be loaded. Please rerun AICLI.")
        return
    general.clear_screen()
    prompt = None
    while prompt is None or prompt == "":
        print("Please enter your prompt:")
        prompt = str(input())
    general.clear_screen()
    print("Generating Text...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **prompt_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(prompt_inputs.input_ids, generated_ids)
    ]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    general.clear_screen()
    print(f"Generated Text:\n===============\n{generated_text}\n")
    input("Press Enter to close the program.")
    return
