import re
from memory import save_preference, get_preference, save_environment_setup, get_environment_setup

def parse_natural_language_task(task_description):
    """Parse a natural language task description and return actionable commands.

    'Set theme to dark and language to English' ->
        [('save_preference', 'theme', 'dark'), ('save_preference', 'language', 'English')]
    Steps are separated by 'and' or 'then'. 'KEY to VALUE' sets a preference (a comma
    list gives a list value), 'enable KEY' sets it to 'enable'. Anything else, including
    conditionals ('if ..., then ...'), is saved as an unrecognized task."""
    unrecognized = [('save_environment_setup', f'unrecognized_task: {task_description}')]
    if re.match(r'\s*if\b', task_description, re.IGNORECASE):
        return unrecognized

    commands = []
    for step in re.split(r'\s+(?:and|then)\s+', task_description.strip(), flags=re.IGNORECASE):
        if match := re.fullmatch(r'(?:set\s+)?(\w[\w ]*?)\s+to\s+(.+)', step, re.IGNORECASE):
            key, value = match.group(1).strip(), match.group(2).strip()
            if ',' in value:
                value = [v.strip() for v in value.split(',')]
            commands.append(('save_preference', key, value))
        elif match := re.fullmatch(r'enable\s+(.+)', step, re.IGNORECASE):
            commands.append(('save_preference', match.group(1).strip(), 'enable'))
    return commands or unrecognized


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