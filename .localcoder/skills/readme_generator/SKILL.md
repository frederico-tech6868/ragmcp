# readme_generator

A skill to generate and maintain the README.md file for the project, ensuring it reflects the project's structure, purpose, and usage guidelines.

## Description

This skill is triggered by phrases like "generate README", "update project documentation", or "write project overview". It creates a structured README.md with sections for project description, features, getting started, contributing, and licensing. The content is derived from the project's codebase, configuration files, and user-provided details.

## Task Flow

1. Analyze the project structure and files to gather relevant information.
2. Generate a README.md with standardized sections and content.
3. Validate the generated README against the project's needs and update if necessary.

## Patterns to Mirror

- Project description: "A project for [purpose]"
- Features list: "- [Feature 1]" "- [Feature 2]"
- Getting started guide: "1. Clone the repository..."
- Contributing instructions: "Contributions are welcome! Please follow the guidelines..."
- License information: "This project is licensed under the [License Name]"