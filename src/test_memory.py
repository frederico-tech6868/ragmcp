import sys
import os
from memory import save_preference, get_preference, save_environment_setup, get_environment_setup

# Test persistent memory storage
if __name__ == "__main__":
    # Save sample preferences
    save_preference("theme", "dark")
    save_preference("language", "en")
    save_preference("font_size", 14)
    
    # Retrieve and verify preferences
    print("Testing persistent memory storage:")
    print(f"Theme: {get_preference('theme', 'default')}")
    print(f"Language: {get_preference('language', 'default')}")
    print(f"Font Size: {get_preference('font_size', 12)}")
    print(f"Non-existent key: {get_preference('nonexistent', 'default')}")
    
    # Test environment setup
    env_setup = {
        "python_version": "3.9.7",
        "dependencies": ["requests", "numpy"]
    }
    save_environment_setup(env_setup)
    
    print("\nEnvironment setup saved successfully.")