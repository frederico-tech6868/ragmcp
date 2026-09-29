import re
import json
from memory import save_preference, get_preference, save_environment_setup, get_environment_setup

def parse_natural_language_task(task_description):
    """Parse a natural language task description and return actionable commands."""
    # Example: 'Set theme to dark and language to English'
    commands = []

    # Extract preferences using regex
    preference_pattern = r'([a-zA-Z_]+)\s+to\s+([a-zA-Z_\s]+)'
    matches = re.findall(preference_pattern, task_description)

    for key, value in matches:
        # Handle cases where value might contain commas or other separators
        # Split on commas if present and store as list
        if ',' in value:
            values = [v.strip() for v in value.split(',')] 
            commands.append(('save_preference', key.strip(), values))
        else:
            commands.append(('save_preference', key.strip(), value.strip()))

    # Handle multi-step commands (e.g., 'then', 'next')
    if 'then' in task_description:
        steps = task_description.split('then')
        for step in steps:
            step_commands = parse_natural_language_task(step.strip())
            commands.extend(step_commands)

    # Handle conditional commands (e.g., 'if', 'then', 'else')
    # Use regex to capture condition, then_action, else_action
    conditional_pattern = r'if\s+(.*?),\s+then\s+(.*?)(?:\s+else\s+(.*?))?'
    match = re.match(conditional_pattern, task_description)
    
    if match:
        condition = match.group(1).strip()
        then_action = match.group(2).strip()
        else_action = match.group(3).strip() if match.group(3) else None
        
        # Parse 'then' action
        then_commands = parse_natural_language_task(f"{condition}, {then_action}")
        commands.extend(then_commands)
        
        # Parse 'else' action if present
        if else_action:
            else_commands = parse_natural_language_task(f"{condition}, {else_action}")
            commands.extend(else_commands)
    
    # Fallback for unrecognized commands
    if not commands:
        # Format as single string for environment setup
        commands.append(('save_environment_setup', f'unrecognized_task: {task_description}'))

    return commands


def execute_parsed_commands(commands):
    """Execute a list of parsed commands."""
    for cmd, *args in commands:
        if cmd == 'save_preference':
            key, value = args
            save_preference(key, value)
        elif cmd == 'save_environment_setup':
            arg = args[0] if args else ''
            save_environment_setup(arg)
        # Add more command types as needed

# Example usage:
#if __name__ == "__main__":
#    task = "Set theme to dark and language to English"
#    commands = parse_natural_language_task(task)
#    execute_parsed_commands(commands)