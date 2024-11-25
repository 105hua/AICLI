import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules import general

def menu():
    model_id = None
    while model_id is None or model_id == "":
        general.clear_screen()
        print("Please enter the ID of the Model you'd like to use:")
        model_id = str(input())
    general.clear_screen()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            load_in_4bit=True,
            token=os.environ["HF_API_KEY"]
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            token=os.environ["HF_API_KEY"]
        )
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