import pytest
from natural_language_parser import parse_natural_language_task

def test_recognized_command():
    task = "Set theme to dark and language to English"
    commands = parse_natural_language_task(task)
    assert commands == [('save_preference', 'theme', 'dark'), ('save_preference', 'language', 'English')]

def test_unrecognized_command():
    task = "Configure the system for high performance"
    commands = parse_natural_language_task(task)
    assert commands == [('save_environment_setup', 'unrecognized_task: Configure the system for high performance')]

def test_mixed_commands():
    task = "Set font size to 14 and enable dark mode"
    commands = parse_natural_language_task(task)
    assert commands == [('save_preference', 'font size', '14'), ('save_preference', 'dark mode', 'enable')]

def test_conditional_command():
    task = "if the system is under load, then set performance mode to high else set to normal"
    commands = parse_natural_language_task(task)
    assert commands == [('save_environment_setup', 'unrecognized_task: if the system is under load, then set performance mode to high else set to normal')]


def test_then_steps_are_not_duplicated():
    commands = parse_natural_language_task("Set theme to dark then set language to English")
    assert commands == [('save_preference', 'theme', 'dark'), ('save_preference', 'language', 'English')]


def test_comma_values_become_a_list():
    commands = parse_natural_language_task("Set languages to English, French")
    assert commands == [('save_preference', 'languages', ['English', 'French'])]
