import sys
import os
from simple_term_menu import TerminalMenu
from modules import general
from menus import image_generation, text_generation

MODELS_PATH = os.path.join(os.getcwd(), "models")
ENV_VARS = ["CIVITAI_API_KEY", "HF_API_KEY"]

def main():
    if not all(env_var in os.environ for env_var in ENV_VARS):
        general.clear_screen()
        print("You have not set all the required environment variables. Please run the program again.")
        return
    os.makedirs(MODELS_PATH, exist_ok=True)
    general.clear_screen()
    options = [
        "Image Generation",
        "Text Generation",
        "Exit"
    ]
    main_terminal_menu = TerminalMenu(
        options,
        title="Welcome to AICLI! Please choose an option:"
    )
    selected_index = main_terminal_menu.show()
    while selected_index is None:
        general.clear_screen()
        print("You have not selected a valid option. Please try again.")
        selected_index = main_terminal_menu.show()
    selected_option = options[selected_index].lower().replace(" ", "_")
    general.clear_screen()
    if selected_option == "exit" or selected_option is None:
        print("AICLI will now exit. See you soon!")
        sys.exit(0)
    if selected_option == "image_generation":
        image_generation.menu()
    if selected_option == "text_generation":
        text_generation.menu()


if __name__ == "__main__":
    main()