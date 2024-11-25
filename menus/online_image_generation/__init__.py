"""
This is the package for the Online Image Generation Menu.
"""

# pylint: disable=E0401,R0914,R0912,R0915,E1136,R0911

import os
from simple_term_menu import TerminalMenu
from openai import OpenAI
from together import Together
import requests
from tqdm import tqdm
from modules import general

def menu():
    """
    This is the menu function for the Online Image Generation Menu.
    :return:
    """
    general.clear_screen()
    service_menu = TerminalMenu(
        [
            "OpenAI",
            "Together AI"
        ],
        title="What service would you like to generate an image on?"
    )
    selected_service_index = service_menu.show()
    while selected_service_index is None:
        general.clear_screen()
        print("You've selected an invalid input, please try again.")
        selected_service_index = service_menu.show()
    if selected_service_index == 0:
        if "OPENAI_API_KEY" not in os.environ:
            general.clear_screen()
            print("Please set the OPENAI_API_KEY variable in your .envrc to use this feature.")
            return
        client = OpenAI()
        general.clear_screen()
        print("Please enter your prompt:")
        prompt = str(input())
        general.clear_screen()
        quality_menu = TerminalMenu(
            [
                "Standard",
                "HD"
            ],
            title="Choose the quality of the image you'd like to generate:"
        )
        selected_quality_index = quality_menu.show()
        while selected_quality_index is None:
            general.clear_screen()
            print("You've selected an invalid input, please try again.")
            selected_quality_index = quality_menu.show()
        if selected_quality_index == 1:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="hd",
                n=1
            )
        else:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
        image_url = response.data[0].url
        try:
            response = requests.get(
                image_url,
                stream=True,
                timeout=30
            )
        except requests.exceptions.Timeout:
            general.clear_screen()
            print("The request timed out, please try again later.")
            return
        if not response.ok:
            general.clear_screen()
            print("Image URL was not valid, please try again.")
            return
        total_size = int(response.headers.get("content-length", 0))
        with open(os.path.join(os.getcwd(), "output.png"), "wb") as file, tqdm(
            desc="output.png",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
        general.clear_screen()
        print("Your image has been generated and saved as 'output.png' in your running directory.")
    elif selected_service_index == 1:
        if "TOGETHER_API_KEY" not in os.environ:
            general.clear_screen()
            print("Please set the TOGETHER_API_KEY variable in your .envrc to use this feature.")
            return
        client = Together()
        model_menu = TerminalMenu(
            [
                "Flux.1 Schnell",
                "Flux.1 Schnell Turbo",
                "Flux.1 Dev",
                "Stable Diffusion XL 1.0"
            ],
            title="Please select the model that you'd like to inference:"
        )
        selected_model_index = model_menu.show()
        while selected_model_index is None:
            general.clear_screen()
            print("You've selected an invalid input, please try again.")
            selected_model_index = model_menu.show()
        match selected_model_index:
            case 0:
                model_str = "black-forest-labs/FLUX.1-schnell-Free"
            case 1:
                model_str = "black-forest-labs/FLUX.1-schnell"
            case 2:
                model_str = "black-forest-labs/FLUX.1-dev"
            case 3:
                model_str = "stabilityai/stable-diffusion-xl-base-1.0"
            case _:
                general.clear_screen()
                print("You've selected an invalid input, please rerun AICLI and try again.")
                return
        general.clear_screen()
        print("Please enter your prompt:")
        prompt = str(input())
        while prompt == "":
            print("Your prompt cannot be empty, please try again.")
            print("Please enter your prompt:")
            prompt = str(input())
        general.clear_screen()
        steps = None
        while steps is None:
            try:
                general.clear_screen()
                print("Please enter the number of steps you'd like to use:")
                steps = int(input())
            except ValueError:
                pass
        general.clear_screen()
        gen_response = client.images.generate(
            model=model_str,
            prompt=prompt,
            steps=steps,
            n=1
        )
        image_url = gen_response.data[0].url
        try:
            response = requests.get(
                image_url,
                stream=True,
                timeout=30
            )
        except requests.exceptions.Timeout:
            general.clear_screen()
            print("The request timed out, please try again later.")
            return
        if not response.ok:
            general.clear_screen()
            print("Image URL was not valid, please try again.")
            return
        total_size = int(response.headers.get("content-length", 0))
        with open(os.path.join(os.getcwd(), "output.png"), "wb") as file, tqdm(
                desc="output.png",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
        general.clear_screen()
        print("Your image has been generated and saved as 'output.png' in your running directory.")
