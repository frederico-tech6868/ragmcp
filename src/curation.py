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


def _curate_one(skill):
    """Evaluate one skill and update it if needed; returns its result entry."""
    skill_name = os.path.basename(str(skill.get('path', '?'))).replace('.md', '').replace('_', ' ')
    entry = {'skill': skill_name, 'timestamp': datetime.now().isoformat()}
    try:
        if not skill.get('metadata'):
            raise KeyError('metadata')
        if evaluate_skill(skill) == 'update':
            update_skill(skill)
            entry['action'] = 'updated'
        else:
            entry['action'] = 'kept'
    except KeyError as e:
        logging.error(f"Error processing skill {skill_name}: Missing metadata {e}")
        entry.update(action='error', error=f"Missing metadata: {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing skill {skill_name}: {e}")
        entry.update(action='error', error=f"{type(e).__name__}: {e}")
    return entry


def curate_skills(skill_data=None):
    """Main function to curate all skills in the system.

    skill_data: a list of skill dicts ({'path', 'content', 'metadata'}); None curates
    every skill in SKILLS_DIR. Each run is appended to the curation log and to the
    curation history in persistent memory; a run that fails as a whole is recorded
    there with an 'error' entry.
    """
    results = []
    try:
        if skill_data is None:
            skill_data = list(load_skills().values())
        os.makedirs(os.path.dirname(CURATION_LOG), exist_ok=True)
        os.makedirs(PERSISTENT_MEMORY_DIR, exist_ok=True)

        results = [_curate_one(skill) for skill in skill_data]
        run = {'timestamp': datetime.now().isoformat(), 'results': results}

        # Log curation results
        log_data = []
        if os.path.exists(CURATION_LOG):
            with open(CURATION_LOG, 'r') as f:
                log_data = json.load(f)
        log_data.append(run)
        with open(CURATION_LOG, 'w') as f:
            json.dump(log_data, f)

        # Save persistent memory: the last run, and the history of all runs
        with open(os.path.join(PERSISTENT_MEMORY_DIR, 'memory.json'), 'w') as f:
            json.dump({'last_curation_time': run['timestamp'], 'results': results}, f)
        save_curation_history(load_curation_history() + [run])

        # Skill creation integration
        if any(result['action'] == 'updated' for result in results):
            # Placeholder: record that skills were updated with a new skill file
            new_skill_content = "# New Skill\n\n## Last Updated\n" + datetime.now().strftime('%Y-%m-%d')
            with open(os.path.join(SKILL_CREATION_DIR, 'new_skill.md'), 'w') as f:
                f.write(new_skill_content)

    except Exception as e:
        # Handle any top-level errors
        logging.critical(f"Critical error in curation process: {e}")
        save_curation_history(load_curation_history() + [
            {'timestamp': datetime.now().isoformat(), 'error': str(e)}])

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
