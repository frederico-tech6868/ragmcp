import os
import json
from datetime import datetime

# Skill creation engine to codify successful task executions into markdown recipes

SKILLS_DIR = 'skills/'
LOG_FILE = 'data/skill_logs.json'

def log_skill(task_name, steps, tools_used, outcome):
    """Log a successful task execution for later codification into a skill."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    skill_log = {
        'timestamp': datetime.now().isoformat(),
        'task_name': task_name,
        'steps': steps,
        'tools_used': tools_used,
        'outcome': outcome
    }
    
    # Append to log file
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(skill_log)
    
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f)


def generate_skill_markdown(task_name):
    """Generate a markdown recipe for a successful task."""
    # This would typically analyze logs and create a structured recipe
    # For now, we'll create a basic template
    markdown = f"""
# {task_name}

## Steps
1. {task_name}

## Tools Used
- {', '.join(['`' + tool + '`' for tool in ['tool1', 'tool2']])}

## Outcome
- Success
"""
    
    return markdown


def save_skill_markdown(task_name, markdown_content):
    """Save a skill as a markdown file."""
    os.makedirs(SKILLS_DIR, exist_ok=True)
    file_path = os.path.join(SKILLS_DIR, f'{task_name.replace(' ', '_')}.md')
    
    with open(file_path, 'w') as f:
        f.write(markdown_content)


def create_skill_from_log(task_name):
    """Create a skill from a logged task execution."""
    # In a real implementation, this would analyze the log and generate a proper recipe
    # Here we'll just create a basic example
    steps = [f"Step {i+1}: {task_name}" for i in range(3)]
    tools_used = ['tool1', 'tool2']
    outcome = 'Success'
    
    markdown = generate_skill_markdown(task_name)
    save_skill_markdown(task_name, markdown)
    
    return markdown