import json
import os

# Persistent memory storage for user preferences and environment setups

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFERENCES_FILE = os.path.join(ROOT, 'data', 'preferences.json')
ENV_SETUP_FILE = os.path.join(ROOT, 'data', 'environment_setup.json')

def save_preference(key, value):
    """Save a user preference to disk."""
    os.makedirs(os.path.dirname(PREFERENCES_FILE), exist_ok=True)
    
    preferences = {}
    if os.path.exists(PREFERENCES_FILE):
        with open(PREFERENCES_FILE, 'r') as f:
            preferences = json.load(f)

    preferences[key] = value
    
    with open(PREFERENCES_FILE, 'w') as f:
        json.dump(preferences, f)


def get_preference(key, default=None):
    """Retrieve a user preference from disk."""
    if os.path.exists(PREFERENCES_FILE):
        with open(PREFERENCES_FILE, 'r') as f:
            return json.load(f).get(key, default)
    return default


def delete_preference(key):
    """Remove a preference key from the preferences file."""
    if not os.path.exists(PREFERENCES_FILE):
        return False

    preferences = {}
    with open(PREFERENCES_FILE, 'r') as f:
        try:
            preferences = json.load(f)
        except json.JSONDecodeError:
            # Handle empty or invalid JSON
            pass

    if key in preferences:
        del preferences[key]
        with open(PREFERENCES_FILE, 'w') as f:
            json.dump(preferences, f)
        return True
    return False

def save_environment_setup(setup_data):
    """Save environment setup information to disk."""
    os.makedirs(os.path.dirname(ENV_SETUP_FILE), exist_ok=True)

    with open(ENV_SETUP_FILE, 'w') as f:
        json.dump(setup_data, f)


def get_environment_setup():
    """Retrieve saved environment setup information."""
    if os.path.exists(ENV_SETUP_FILE):
        with open(ENV_SETUP_FILE, 'r') as f:
            return json.load(f)
    return {}
