import os
from simple_term_menu import TerminalMenu
from modules import general
from menus import (
    local_text_generation,
    local_image_generation,
    online_image_generation,
    online_text_generation,
    gpu_info
)

MODELS_PATH = os.path.join(os.getcwd(), "models")

def main():
    os.makedirs(MODELS_PATH, exist_ok=True)
    general.clear_screen()
    options = [
        "Local Image Generation",
        "Local Text Generation",
        "Online Image Generation",
        "Online Text Generation",
        "GPU Info",
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
        return
    elif selected_option == "local_image_generation":
        local_image_generation.menu()
    elif selected_option == "local_text_generation":
        local_text_generation.menu()
    elif selected_option == "online_image_generation":
        online_image_generation.menu()
    elif selected_option == "online_text_generation":
        online_text_generation.menu()
    elif selected_option == "gpu_info":
        gpu_info.menu()


if __name__ == "__main__":
    main()