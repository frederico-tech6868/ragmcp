import json
import os

# Persistent memory storage for user preferences and environment setups

PREFERENCES_FILE = 'data/preferences.json'
ENV_SETUP_FILE = 'data/environment_setup.json'

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