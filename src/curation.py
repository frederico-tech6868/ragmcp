"""Autonomous curation: evaluate skills, update them, and log the results."""

import glob
import json
import logging
import os
import threading
import time
from datetime import datetime

# Paths are relative to the project root, whatever the current directory is.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = os.path.join(ROOT, 'skills')
SKILL_CREATION_DIR = os.path.join(ROOT, 'skills')
CURATION_LOG = os.path.join(ROOT, 'data', 'curation_log.json')
PERSISTENT_MEMORY_DIR = os.path.join(ROOT, 'data', 'persistent_memory')

# Background execution loop configuration
CURATION_INTERVAL = 60  # 60 seconds for faster feedback

os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)
logging.basicConfig(
    filename=os.path.join(ROOT, 'data', 'curation_debug.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Skill creation logic
def generate_skill_content(skill_name, metadata):
    """Generate new skill content based on metadata."""
    content = f"# {skill_name}\n\n## Description\n{metadata.get('description', '')}\n\n## Parameters\n{metadata.get('parameters', '')}\n\n## Example\n{metadata.get('example', '')}\n\n## Last Updated\n{datetime.now().strftime('%Y-%m-%d')}\n"
    return content


def create_new_skill(skill_name, metadata):
    """Create a new skill file with generated content."""
    content = generate_skill_content(skill_name, metadata)
    skill_path = os.path.join(SKILL_CREATION_DIR, f'{skill_name.replace(' ', '_')}.md')
    try:
        with open(skill_path, 'w') as f:
            f.write(content)
        logging.info(f"Created new skill: {skill_path}")
        return skill_path
    except Exception as e:
        logging.error(f"Failed to create new skill {skill_name}: {e}")
        return None

# Persistent memory functions
def save_curation_history(history):
    """Save curation history to persistent memory."""
    try:
        os.makedirs(PERSISTENT_MEMORY_DIR, exist_ok=True)
        with open(os.path.join(PERSISTENT_MEMORY_DIR, 'curation_history.json'), 'w') as f:
            json.dump(history, f)
        logging.info("Saved curation history to persistent memory")
    except Exception as e:
        logging.error(f"Failed to save curation history: {e}")
        return None

def load_curation_history():
    """Load curation history from persistent memory."""
    try:
        history_file = os.path.join(PERSISTENT_MEMORY_DIR, 'curation_history.json')
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                return json.load(f)
        else:
            return []
    except Exception as e:
        logging.error(f"Failed to load curation history: {e}")
        return []


def load_skills():
    """Load all skills from the skills directory."""
    skills = {}
    for filepath in glob.glob(os.path.join(SKILLS_DIR, '*.md')):
        filename = os.path.basename(filepath)
        skill_name = filename.replace('.md', '').replace('_', ' ')

        # Parse metadata from the markdown file
        with open(filepath, 'r') as f:
            content = f.read()

        # Simple metadata parsing (would be more complex in production)
        metadata = {}
        if '## Last Updated' in content:
            metadata['last_updated'] = content.split('## Last Updated')[1].split('\n')[0].strip()

        skills[skill_name] = {
            'path': filepath,
            'content': content,
            'metadata': metadata
        }

    return skills


def evaluate_skill(skill_data):
    """Evaluate a skill based on performance metrics."""
    # In a real implementation, this would analyze task success rates,
    # execution time, and other metrics

    # Evaluate skill based on execution count and error rate
    if 'execution_count' in skill_data['metadata'] and 'error_rate' in skill_data['metadata']:
        execution_count = skill_data['metadata']['execution_count']
        error_rate = skill_data['metadata']['error_rate']

        # Update if execution count is low or error rate is high
        if execution_count < 5 or error_rate > 0.2:
            return 'update'

    return 'keep'


def update_skill(skill_data):
    """Update a skill with new information or improvements."""
    # In a real implementation, this would involve:
    # 1. Analyzing recent task executions
    # 2. Generating improved steps/tools
    # 3. Updating the markdown file

    # For now, we'll just add a curation note
    new_content = skill_data['content'] + '\n\n## Curated on ' + datetime.now().strftime('%Y-%m-%d') + '\n\n* Updated based on recent performance metrics'
    with open(skill_data['path'], 'w') as f:
        f.write(new_content)

    return True


def curate_skills(skill_data=None):
    """Main function to curate all skills in the system.

    skill_data: a list of skill dicts ({'path', 'content', 'metadata'}); None curates
    every skill in SKILLS_DIR.
    """
    if skill_data is None:
        skill_data = list(load_skills().values())
    results = []
    try:
        os.makedirs(os.path.dirname(CURATION_LOG), exist_ok=True)
        os.makedirs(PERSISTENT_MEMORY_DIR, exist_ok=True)

        results = []

        for skill in (skill_data or []):
            try:
                skill_name = skill['path'].replace('.md', '').replace('_', ' ')
                skill_data = skill

                status = evaluate_skill(skill_data)
                if status == 'update':
                    updated = update_skill(skill_data)
                    results.append({
                        'skill': skill_name,
                        'action': 'updated',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    results.append({
                        'skill': skill_name,
                        'action': 'kept',
                        'timestamp': datetime.now().isoformat()
                    })
            except KeyError as e:
                # Handle missing metadata in skill data
                logging.error(f"Error processing skill {skill_name}: Missing metadata {e}")
                results.append({
                    'skill': skill_name,
                    'action': 'error',
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                })
            except Exception as e:
                # Handle other unexpected errors
                logging.error(f"Unexpected error processing skill {skill_name}: {e}")
                results.append({
                    'skill': skill_name,
                    'action': 'error',
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                })

        # Log curation results
        if os.path.exists(CURATION_LOG):
            with open(CURATION_LOG, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = []

        log_data.append({
            'timestamp': datetime.now().isoformat(),
            'results': results
        })

        with open(CURATION_LOG, 'w') as f:
            json.dump(log_data, f)

        # Save persistent memory
        persistent_memory = {
            'last_curation_time': datetime.now().isoformat(),
            'results': results
        }

        with open(os.path.join(PERSISTENT_MEMORY_DIR, 'memory.json'), 'w') as f:
            json.dump(persistent_memory, f)

        # Skill creation integration
        if any(result['action'] == 'updated' for result in results):
            # Trigger skill creation process for updated skills
            # This is a placeholder - actual implementation would involve
            # generating new skill content and saving it to SKILL_CREATION_DIR
            # For demonstration, we'll create a new skill file
            new_skill_content = "# New Skill\n\n## Last Updated\n" + datetime.now().strftime('%Y-%m-%d')
            new_skill_path = os.path.join(SKILL_CREATION_DIR, 'new_skill.md')
            with open(new_skill_path, 'w') as f:
                f.write(new_skill_content)

    except Exception as e:
        # Handle any top-level errors
        logging.critical(f"Critical error in curation process: {e}")
        with open(os.path.join(PERSISTENT_MEMORY_DIR, 'memory.json'), 'w') as f:
            json.dump({'error': str(e)}, f)

    return results


def start_curation_loop():
    """Start the background curation loop."""
    def loop():
        while True:
            curate_skills()
            time.sleep(CURATION_INTERVAL)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread
