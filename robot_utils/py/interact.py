from pathlib import Path
from typing import Union, List, Any
from robot_utils import console
from rich import print
from rich.panel import Panel
import inquirer


def ask_list(message: str, choices: list) -> Union[str, Path]:
    question = [inquirer.List("question", message=message, choices=choices)]
    return inquirer.prompt(question)["question"]


def ask_binary(message: str) -> Union[str, Path]:
    question = [inquirer.List("question", message=message, choices=["Yes", "No"])]
    return inquirer.prompt(question)["question"] == "Yes"


def ask_checkbox(message: str, choices: List[Any]) -> Union[List[Any]]:
    question = [inquirer.Checkbox("question", message=message, choices=choices)]
    return inquirer.prompt(question)["question"]


def ask_checkbox_with_all(message: str, choices: list) -> Union[List[Any]]:
    selected = []
    while len(selected) < 1:
        selected.extend(ask_checkbox(message, ["all"] + choices))

    if len(selected) == 1 and selected[0] == "all":
        selected = choices

    return selected


def ask_text(message: str):
    question = [inquirer.Text("question", message=message)]
    return inquirer.prompt(question)["question"]


def user_warning(text):
    # padding = " " * ((console.width - len(text)) // 2)
    # text = padding + text + padding
    # box_content = Panel.fit(text, title="Warning", style='bold red', border_style="red")
    # print(box_content)
    box_message(text, "red", "red", "Warning")


def box_message(text: str, color: str, border_color: str = "", title: str = ""):
    padding = " " * ((console.width - len(text)) // 2)
    text = padding + text + padding
    box_content = Panel.fit(text, title=title, style=f'bold {color}', border_style=f"{border_color}")
    print(box_content)


if __name__ == "__main__":
    user_warning("Hello, this is a centered box!")