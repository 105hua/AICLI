"""
This is the package for the Online Text Generation Menu.
"""

# pylint: disable=C0301,E0401,R0915

import os
from openai import OpenAI
from together import Together
from simple_term_menu import TerminalMenu
from modules import general

OPENAI_OPTIONS = [
    {"name": "GPT-4o", "value": "gpt-4o"},
    {"name": "GPT-4o Mini", "value": "gpt-4o-mini"},
    {"name": "o1 Preview", "value": "o1-preview"},
    {"name": "o1 Mini", "value": "o1-mini"}
]

TOGETHER_OPTIONS = [
    {"name": "Llama 3.1 8B Instruct Turbo", "value": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"},
    {"name": "Llama 3.1 70B Instruct Turbo", "value": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"},
    {"name": "Llama 3.1 405B Instruct Turbo", "value": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"},
    {"name": "Llama 3 8B Instruct Turbo", "value": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo"},
    {"name": "Llama 3 70B Instruct Turbo", "value": "meta-llama/Meta-Llama-3-70B-Instruct-Turbo"},
    {"name": "Llama 3.2 3B Instruct Turbo", "value": "meta-llama/Llama-3.2-3B-Instruct-Turbo"},
    {"name": "Llama 3 8B Instruct Lite", "value": "meta-llama/Meta-Llama-3-8B-Instruct-Lite"},
    {"name": "Llama 3 70B Instruct Lite", "value": "meta-llama/Meta-Llama-3-70B-Instruct-Lite"},
    {"name": "Llama 3 8B Instruct Reference", "value": "meta-llama/Llama-3-8b-chat-hf"},
    {"name": "Llama 3 70B Instruct Reference", "value": "meta-llama/Llama-3-70b-chat-hf"},
    {"name": "Llama 3.1 Nemotron 70B", "value": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"},
    {"name": "Qwen 2.5 Coder 32B Instruct", "value": "Qwen/Qwen2.5-Coder-32B-Instruct"},
    {"name": "WizardLM-2 8x22B", "value": "microsoft/WizardLM-2-8x22B"},
    {"name": "Gemma 2 27B", "value": "google/gemma-2-27b-it"},
    {"name": "Gemma 2 9B", "value": "google/gemma-2-9b-it"},
    {"name": "DBRX Instruct", "value": "databricks/dbrx-instruct"},
    {"name": "DeepSeek LLM Chat (67B)", "value": "deepseek-ai/deepseek-llm-67b-chat"},
    {"name": "Gemma Instruct (2B)", "value": "google/gemma-2b-it"},
    {"name": "MythoMax-L2 (13B)", "value": "Gryphe/MythoMax-L2-13b"},
    {"name": "LLaMA-2 Chat (13B)", "value": "meta-llama/Llama-2-13b-chat-hf"},
    {"name": "Mistral (7B) Instruct", "value": "mistralai/Mistral-7B-Instruct-v0.1"},
    {"name": "Mistral (7B) Instruct v0.2", "value": "mistralai/Mistral-7B-Instruct-v0.2"},
    {"name": "Mistral (7B) Instruct v0.3", "value": "mistralai/Mistral-7B-Instruct-v0.3"},
    {"name": "Mixtral-8x7B Instruct (46.7B)", "value": "mistralai/Mixtral-8x7B-Instruct-v0.1"},
    {"name": "Mixtral-8x22B Instruct (141B)", "value": "mistralai/Mixtral-8x22B-Instruct-v0.1"},
    {"name": "Nous Hermes 2 - Mixtral 8x7B-DPO (46.7B)", "value": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"},
    {"name": "Qwen 2.5 7B Instruct Turbo", "value": "Qwen/Qwen2.5-7B-Instruct-Turbo"},
    {"name": "Qwen 2.5 72B Instruct Turbo", "value": "Qwen/Qwen2.5-72B-Instruct-Turbo"},
    {"name": "Qwen 2 Instruct (72B)", "value": "Qwen/Qwen2-72B-Instruct"},
    {"name": "StripedHyena Nous (7B)", "value": "togethercomputer/StripedHyena-Nous-7B"},
    {"name": "Upstage SOLAR Instruct v1 (11B)", "value": "upstage/SOLAR-10.7B-Instruct-v1.0"}
]

def menu():
    """
    This is the Menu for Online Text Generation.
    :return:
    """
    general.clear_screen()
    service_menu = TerminalMenu(
        [
            "OpenAI",
            "Together AI"
        ],
        title="What service would you like to use?"
    )
    selected_service_index = service_menu.show()
    while selected_service_index is None:
        general.clear_screen()
        print("You've selected an invalid input, please try again.")
        selected_service_index = service_menu.show()
    if selected_service_index == 0:
        if "OPENAI_API_KEY" not in os.environ:
            general.clear_screen()
            print("Please ensure the OPENAI_API_KEY variable is set to use this feature.")
            return
        general.clear_screen()
        client = OpenAI()
        model_menu = TerminalMenu(
            [option["name"] for option in OPENAI_OPTIONS],
            title="Please select the model that you'd like to use:"
        )
        selected_model_index = model_menu.show()
        while selected_model_index is None:
            general.clear_screen()
            print("You've selected an invalid input, please try again.")
            selected_model_index = model_menu.show()
        selected_model = OPENAI_OPTIONS[selected_model_index]["value"]
        general.clear_screen()
        print("Please enter your prompt:")
        prompt = str(input())
        while prompt == "":
            print("Your prompt cannot be empty, please try again.")
            print("Please enter your prompt:")
            prompt = str(input())
        general.clear_screen()
        completion = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}]
        )
        response = completion.choices[0].message.content
        general.clear_screen()
        print(f"Generated Text\n====================\n\n{response}\n")
    elif selected_service_index == 1:
        if "TOGETHER_API_KEY" not in os.environ:
            general.clear_screen()
            print("Please ensure the TOGETHER_API_KEY variable is set to use this feature.")
            return
        general.clear_screen()
        client = Together()
        model_menu = TerminalMenu(
            [option["name"] for option in TOGETHER_OPTIONS],
            title="Please select the model that you'd like to use:"
        )
        selected_model_index = model_menu.show()
        while selected_model_index is None:
            general.clear_screen()
            print("You've selected an invalid input, please try again.")
            selected_model_index = model_menu.show()
        selected_model = TOGETHER_OPTIONS[selected_model_index]["value"]
        general.clear_screen()
        print("Please enter your prompt:")
        prompt = str(input())
        while prompt == "":
            general.clear_screen()
            print("Your prompt cannot be empty, please try again.")
            print("Please enter your prompt:")
            prompt = str(input())
        general.clear_screen()
        completion = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}]
        )
        response = completion.choices[0].message.content
        general.clear_screen()
        print(f"Generated Text\n====================\n\n{response}\n")
