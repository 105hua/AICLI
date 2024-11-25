import os
from openai import OpenAI
from together import Together
from simple_term_menu import TerminalMenu
from modules import general

OPENAI_OPTIONS = [
    {
        "name": "GPT-4o",
        "value": "gpt-4o"
    },
    {
        "name": "GPT-4o Mini",
        "value": "gpt-4o-mini"
    },
    {
        "name": "o1 Preview",
        "value": "o1-preview"
    },
    {
        "name": "o1 Mini",
        "value": "o1-mini"
    }
]

def menu():
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