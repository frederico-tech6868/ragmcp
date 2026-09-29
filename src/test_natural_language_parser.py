import sys
import os
from natural_language_parser import parse_natural_language_task, execute_parsed_commands

# Test natural language parsing
if __name__ == "__main__":
    # Test case 1: Recognized command
    task1 = "Set theme to dark and language to English"
    commands1 = parse_natural_language_task(task1)
    print("Test 1 - Recognized command:")
    print(commands1)
    execute_parsed_commands(commands1)

    # Test case 2: Unrecognized command
    task2 = "Configure the system for high performance"
    commands2 = parse_natural_language_task(task2)
    print("\nTest 2 - Unrecognized command:")
    print(commands2)
    execute_parsed_commands(commands2)

    # Test case 3: Mixed commands
    task3 = "Set font size to 14 and enable dark mode"
    commands3 = parse_natural_language_task(task3)
    print("\nTest 3 - Mixed commands:")
    print(commands3)
    execute_parsed_commands(commands3)

    # Test case 4: Conditional command
    task4 = "if the system is under load, then set performance mode to high else set to normal"
    commands4 = parse_natural_language_task(task4)
    print("\nTest 4 - Conditional command:")
    print(commands4)
    execute_parsed_commands(commands4)