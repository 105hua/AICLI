import sys
import os
import json
from simple_term_menu import TerminalMenu
import diffusers
import torch
import requests
from tqdm import tqdm
from modules import general

STABLE_DIFFUSION_MODELS = [
    "SD 1.4",
    "SD 1.5",
    "SD 1.5 LCM",
    "SD 1.5 Hyper",
    "SD 2.0",
    "SD 2.1"
]

STABLE_DIFFUSION_THREE_MODELS = [
    "SD 3",
    "SD 3.5",
    "SD 3.5 Medium",
    "SD 3.5 Large",
    "SD 3.5 Large Turbo"
]

SDXL_MODELS = [
    "SDXL 1.0",
    "Pony",
    "SDXL 1.0 LCM",
    "SDXL Turbo",
    "SDXL Lightning",
    "SDXL Hyper",
    "Illustrious"
]

FLUX_MODELS = [
    "Flux .1 D",
    "Flux .1 S"
]

DTYPE_OPTIONS = [
        {
            "name": "Float 16",
            "dtype": torch.float16
        },
        {
            "name": "Float 32",
            "dtype": torch.float32
        },
        {
            "name": "Brain Float 16",
            "dtype": torch.bfloat16
        }
]

MODELS_PATH = os.path.join(os.getcwd(), "models")

def _hf_inference():
    # Ask for a Model ID.
    model_id = None
    while model_id is None or model_id == "":
        general.clear_screen()
        print("What is the ID of the Model you'd like to use?")
        model_id = str(input())
    # Ask for a Model Data Type.
    general.clear_screen()
    dtype_menu = TerminalMenu(
        [option["name"] for option in DTYPE_OPTIONS],
        title="What data type would you like to use?"
    )
    selected_dtype_index = dtype_menu.show()
    while selected_dtype_index is None:
        general.clear_screen()
        print("You have not selected a valid option.")
        selected_dtype_index = dtype_menu.show()
    dtype = DTYPE_OPTIONS[selected_dtype_index]["dtype"]
    # Ask the user if they want the model quantized.
    general.clear_screen()
    quantized_menu = TerminalMenu(
        ["Yes", "No"],
        title="If you are running a variant of Stable Diffusion 3.5 on a limited amount of VRAM, you may find it "
              "beneficial to drop its Text Encoder for reduced memory intensity, with only a slight loss in "
              "performance. Would you like to do so?"
    )
    selected_quantized_index = quantized_menu.show()
    while selected_quantized_index is None:
        general.clear_screen()
        print("You have not selected a valid option.")
        selected_quantized_index = quantized_menu.show()
    if selected_quantized_index == 0:
        try:
            pipe = diffusers.AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=dtype,
                text_encoder_3=None,
                tokenizer_3=None
            )
        except OSError:
            general.clear_screen()
            print("The pipeline could not be created. Please ensure you have entered the Model ID correctly and "
                  "try again.")
            sys.exit(0)
    else:
        try:
            pipe = diffusers.AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=dtype
            )
        except OSError:
            general.clear_screen()
            print("The pipeline could not be created. Please ensure you have entered the Model ID correctly and "
                  "try again.")
            sys.exit(0)
    # Apply other optimisations
    pipe.enable_model_cpu_offload()
    # Ask for prompt.
    prompt = None
    while prompt is None or prompt == "":
        general.clear_screen()
        print("Enter your Positive Prompt:")
        prompt = str(input())
    # Ask if the program should load a negative prompt from a txt file or allow the user to enter it themselves.
    general.clear_screen()
    file_or_input_menu = TerminalMenu(
        ["File", "Input"],
        title="Would you like to load a Negative Prompt from a File or enter it yourself?"
    )
    file_or_input_index = file_or_input_menu.show()
    negative_prompt = ""
    while file_or_input_index is None:
        general.clear_screen()
        print("You chose an invalid input. Please try again.")
        file_or_input_index = file_or_input_menu.show()
    if file_or_input_index == 0:
        general.clear_screen()
        print("Please drop your Negative Prompt Text File into the directory you are running AICLI from "
              "and enter its name here:")
        file_name = str(input())
        file_path = os.path.join(os.getcwd(), file_name)
        if not os.path.exists(file_path):
            general.clear_screen()
            print("That file does not exist in this directory, please rerun the program and try again.")
            return
        with open(file_path, "r", encoding="UTF-8") as f:
            negative_prompt = f.read()
    elif file_or_input_index == 1:
        general.clear_screen()
        print("Enter your Negative Prompt:")
        negative_prompt = str(input())
    # Ask for the width of the image.
    width = None
    while width is None:
        try:
            general.clear_screen()
            print("Enter the Width of your image in px:")
            width = int(input())
        except ValueError:
            pass
    # Ask for the height of the image.
    height = None
    while height is None:
        try:
            general.clear_screen()
            print("Enter the Height of your image in px:")
            height = int(input())
        except ValueError:
            pass
    # Ask for the steps of the image.
    steps = None
    while steps is None:
        try:
            general.clear_screen()
            print("Enter how many Steps you'd like to use:")
            steps = int(input())
        except ValueError:
            pass
    # Ask for the CFG Scale for the image.
    cfg_scale = None
    while cfg_scale is None:
        try:
            general.clear_screen()
            print("Enter the CFG Scale you'd like to use (Usually between 4.0 and 10.0):")
            cfg_scale = float(input())
        except ValueError:
            pass
    # Overview of selected options:
    general.clear_screen()
    print(
        "Configuration Overview\n"
        "======================\n"
        f"Model: {model_id}\n"
        f"Quantized: {'Yes' if selected_quantized_index == 0 else 'No'}\n"
        f"Prompt: {f'{prompt[:100]}...' if len(prompt) > 100 else prompt}\n"
        f"Negative Prompt: {f'{negative_prompt[:100]}...' if len(negative_prompt) > 100 else negative_prompt}\n"
        f"Width: {width}px\n"
        f"Height: {height}px\n"
        f"Steps: {steps}\n"
        f"CFG Scale: {cfg_scale}\n\n"
        "Press Enter to begin the inference."
    )
    input()
    print("\n")
    # Inference the Model
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg_scale
    ).images[0]
    # Save the image.
    image.save(os.path.join(os.getcwd(), "output.png"))
    print("\nYour image has now been generated! You can find it saved as 'output.png' in your running directory.")

def _civitai_inference():
    # Ask the user to choose a model.
    cf_files = [file for file in os.listdir(MODELS_PATH) if file.endswith(".json")]
    st_files = [file for file in os.listdir(MODELS_PATH) if file.endswith(".safetensors")]
    model_files = []
    if len(cf_files) == 0 or len(st_files) == 0:
        print("No models have been downloaded.")
        return
    for cf_file in cf_files:
        st_name = cf_file.replace(".json", ".safetensors")
        if st_name in st_files:
            model_files.append({"config": cf_file, "st": st_name})
    model_names = []
    for model in model_files:
        with open(f"./models/{model['config']}", "r", encoding="UTF-8") as cf:
            config = json.loads(cf.read())
        model_names.append(config["name"])
    general.clear_screen()
    model_menu = TerminalMenu(
        model_names,
        title="Choose a Model to use:"
    )
    selected_model_index = model_menu.show()
    while selected_model_index is None:
        general.clear_screen()
        print("You've selected an invalid input, please try again.")
        selected_model_index = model_menu.show()
    selected_model = model_files[selected_model_index]
    with open(f"./models/{selected_model['config']}", "r", encoding="UTF-8") as cf:
        model_config = json.loads(cf.read())
    # Ask the user for a data type.
    general.clear_screen()
    dtype_menu = TerminalMenu(
        [option["name"] for option in DTYPE_OPTIONS],
        title="Choose the Data Type for your Model:"
    )
    selected_dtype_index = dtype_menu.show()
    while selected_dtype_index is None:
        general.clear_screen()
        print("You have not selected a valid option.")
        selected_dtype_index = dtype_menu.show()
    selected_dtype = DTYPE_OPTIONS[selected_dtype_index]["dtype"]
    # Create the pipe.
    pipe = None
    match config["pipeline"]:
        case "sd":
            pipe = diffusers.StableDiffusionPipeline.from_single_file(
                model_config["model_path"],
                torch_dtype=selected_dtype
            )
        case "sd3":
            pipe = diffusers.StableDiffusion3Pipeline.from_single_file(
                model_config["model_path"],
                torch_dtype=selected_dtype
            )
        case "sdxl":
            pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
                model_config["model_path"],
                torch_dtype=selected_dtype
            )
        case "flux":
            pipe = diffusers.FluxPipeline.from_single_file(
                model_config["model_path"],
                torch_dtype=selected_dtype
            )
        case _:
            print("Pipeline not recognised.")
            return
    if pipe is None:
        print("Pipe was not assigned properly.")
        return
    pipe.enable_model_cpu_offload()
    # Ask for prompt.
    prompt = None
    while prompt is None or prompt == "":
        general.clear_screen()
        print("Enter your Positive Prompt:")
        prompt = str(input())
    # Ask if the program should load a negative prompt from a txt file or allow the user to enter it themselves.
    general.clear_screen()
    file_or_input_menu = TerminalMenu(
        ["File", "Input"],
        title="Would you like to load a Negative Prompt from a File or enter it yourself?"
    )
    file_or_input_index = file_or_input_menu.show()
    negative_prompt = ""
    while file_or_input_index is None:
        general.clear_screen()
        print("You chose an invalid input. Please try again.")
        file_or_input_index = file_or_input_menu.show()
    if file_or_input_index == 0:
        general.clear_screen()
        print("Please drop your Negative Prompt Text File into the directory you are running AICLI from "
              "and enter its name here:")
        file_name = str(input())
        file_path = os.path.join(os.getcwd(), file_name)
        if not os.path.exists(file_path):
            general.clear_screen()
            print("That file does not exist in this directory, please rerun the program and try again.")
            return
        with open(file_path, "r", encoding="UTF-8") as f:
            negative_prompt = f.read()
    elif file_or_input_index == 1:
        general.clear_screen()
        print("Enter your Negative Prompt:")
        negative_prompt = str(input())
    # Ask for the width of the image.
    width = None
    while width is None:
        try:
            general.clear_screen()
            print("Enter the Width of your image in px:")
            width = int(input())
        except ValueError:
            pass
    # Ask for the height of the image.
    height = None
    while height is None:
        try:
            general.clear_screen()
            print("Enter the Height of your image in px:")
            height = int(input())
        except ValueError:
            pass
    # Ask for the steps of the image.
    steps = None
    while steps is None:
        try:
            general.clear_screen()
            print("Enter how many Steps you'd like to use:")
            steps = int(input())
        except ValueError:
            pass
    # Ask for the CFG Scale for the image.
    cfg_scale = None
    while cfg_scale is None:
        try:
            general.clear_screen()
            print("Enter the CFG Scale you'd like to use (Usually between 4.0 and 10.0):")
            cfg_scale = float(input())
        except ValueError:
            pass
    # Ask for the scheduler for the model.
    general.clear_screen()
    scheduler_menu = TerminalMenu(
        [
            f"Default ({pipe.scheduler.__class__.__name__})",
            "Euler Discrete",
            "Euler Discrete with V-Prediction",
            "Euler Ancestral",
            "DDIM",
            "PNDM"
        ],
        title="Choose a scheduler:"
    )
    selected_scheduler_index = scheduler_menu.show()
    while selected_scheduler_index is None:
        general.clear_screen()
        print("You have not selected a valid option.")
        selected_scheduler_index = scheduler_menu.show()
    match selected_scheduler_index:
        case 0:
            pass
        case 1:
            pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
                pipe.scheduler.config
            )
        case 2:
            pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
                pipe.scheduler.config,
                **{"prediction_type": "v_prediction", "rescale_betas_zero_snr": True}
            )
        case 3:
            pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
                pipe.scheduler.config
            )
        case 4:
            pipe.scheduler = diffusers.DDIMScheduler.from_config(
                pipe.scheduler.config
            )
        case 5:
            pipe.scheduler = diffusers.PNDMScheduler.from_config(
                pipe.scheduler.config
            )
        case _:
            pass
    # Overview of selected options:
    general.clear_screen()
    print(
        "Configuration Overview\n"
        "======================\n"
        f"Model: {model_names[selected_model_index]}\n"
        f"Prompt: {f'{prompt[:100]}...' if len(prompt) > 100 else prompt}\n"
        f"Negative Prompt: {f'{negative_prompt[:100]}...' if len(negative_prompt) > 100 else negative_prompt}\n"
        f"Width: {width}px\n"
        f"Height: {height}px\n"
        f"Steps: {steps}\n"
        f"CFG Scale: {cfg_scale}\n\n"
        "Press Enter to begin the inference."
    )
    input()
    print("\n")
    # Inference the Model
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg_scale
    ).images[0]
    # Save the image.
    image.save(os.path.join(os.getcwd(), "output.png"))
    print("\nYour image has now been generated! You can find it saved as 'output.png' in your running directory.")

def _civitai_download():
    if "CIVITAI_API_KEY" not in os.environ:
        general.clear_screen()
        print("Please set the CIVITAI_API_KEY variable in your .envrc to use this feature.")
        return
    model_id = None
    while model_id is None:
        try:
            general.clear_screen()
            print("Please enter your Model ID:")
            model_id = int(input())
        except ValueError:
            pass
    response = requests.get(
        f"https://civitai.com/api/v1/models/{model_id}",
        headers={"Authorization": f"Bearer {os.environ['CIVITAI_API_KEY']}"}
    )
    if not response.ok:
        general.clear_screen()
        print("We couldn't get any information on that Model from CivitAI. Are you sure you have the right ID?")
        return
    data = response.json()
    if "error" in data:
        general.clear_screen()
        print("The CivitAI API returned an error in your request. This is likely due to an Invalid ID.")
        return
    if data["type"].lower() != "checkpoint":
        general.clear_screen()
        print("Please ensure that you are downloading a model that is listed as a Checkpoint.")
        return
    model_name = data["name"]
    latest_version = data["modelVersions"][0]
    file_choices = []
    for file in latest_version["files"]:
        file_choices.append(
            f"{file['metadata']['size'].title()} Model | "
            f"{file['metadata']['fp'].upper()} | "
            f"{file['metadata']['format']}"
        )
    file_menu = TerminalMenu(
        file_choices,
        title=f"Choose a file to use for '{model_name}'"
    )
    selected_file_index = file_menu.show()
    while selected_file_index is None:
        general.clear_screen()
        print("You've selected an invalid input, please try again.")
        selected_file_index = file_menu.show()
    selected_file = latest_version["files"][selected_file_index]
    file_url = selected_file["downloadUrl"]
    file_name = selected_file["name"]
    file_path = os.path.join(MODELS_PATH, file_name)
    base_model = latest_version["baseModel"]
    model_type = "na"
    if base_model in STABLE_DIFFUSION_MODELS:
        model_type = "sd",
    elif base_model in STABLE_DIFFUSION_THREE_MODELS:
        model_type = "sd3",
    elif base_model in SDXL_MODELS:
        model_type = "sdxl"
    elif base_model in FLUX_MODELS:
        model_type = "flux"
    if model_type == "na":
        general.clear_screen()
        print("The Base Model for this Model is not supported by AICLI.")
        return
    download_response = requests.get(
        file_url,
        headers={"Authorization": f"Bearer {os.environ['CIVITAI_API_KEY']}"},
        stream=True
    )
    if not download_response.ok:
        general.clear_screen()
        print("Download URL response was not OK. Please rerun AICLI and try again.")
        return
    general.clear_screen()
    total_size = int(download_response.headers.get("content-length", 0))
    with open(file_path, "wb") as file, tqdm(
        desc=file_name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for data in download_response.iter_content(1024):
            bar.update(len(data))
            file.write(data)
    config_json = {
        "name": model_name,
        "model_path": f"./models/{file_name}",
        "pipeline": model_type
    }
    with open(f"./models/{file_name.replace('.safetensors', '.json')}", "w", encoding="UTF-8") as json_f:
        json_f.write(json.dumps(config_json, indent=4))
    general.clear_screen()
    print("The Model has been downloaded successfully!")

def _civitai_submenu():
    general.clear_screen()
    civitai_menu = TerminalMenu(
        [
            "Download a Model",
            "Inference"
        ],
        title="Choose an option:"
    )
    selected_option_index = civitai_menu.show()
    while selected_option_index is None:
        general.clear_screen()
        print("You've entered an invalid input, please try again")
        selected_option_index = civitai_menu.show()
    if selected_option_index == 0:
        _civitai_download()
    elif selected_option_index == 1:
        _civitai_inference()
    else:
        print("An invalid index has been selected.")

def menu():
    # Ask the user if they want an online inference or a local inference.
    inference_type_menu = TerminalMenu(
        [
            "HuggingFace",
            "CivitAI"
        ],
        title="What service is the model you'd like to use hosted on?"
    )
    selected_index = inference_type_menu.show()
    while selected_index is None:
        general.clear_screen()
        print("You have not selected a valid option. Please try again.")
        selected_index = inference_type_menu.show()
    # Run the relevant function
    if selected_index == 0:
        _hf_inference()
    elif selected_index == 1:
        _civitai_submenu()
    else:
        general.clear_screen()
        print("An invalid option was selected. Please run the program again.")
        return