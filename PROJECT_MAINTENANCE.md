# Project Maintenance Guide

This guide outlines the process for maintaining and updating the `async_llm_handler` project, including version management and releases.

## Project Structure

The project uses the following files for version management and releases:

- `pyproject.toml`: Project configuration and metadata
- `version.txt`: Contains the current version number
- `update.ps1`: PowerShell script for automating the release process

## Updating the Project

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- pip
- git

Install the required tools:

```powershell
pip install build twine
```

### Making Changes

1. Make your desired changes to the code, README, or other files.
2. Test your changes thoroughly.

### Releasing a New Version

1. Open PowerShell in your project directory.
2. Run the update script:

   ```powershell
   .\update.ps1
   ```

3. When prompted, enter the new version number (e.g., 0.1.1).
4. The script will automatically:
   - Update `version.txt`
   - Update version references in `README.md`
   - Commit changes to git
   - Create a new git tag
   - Push changes and tags to GitHub
   - Build the Python package
   - Upload the new version to PyPI

## Manual Steps (if needed)

If you need to perform any steps manually:

### Updating version.txt

1. Open `version.txt`
2. Change the version number
3. Save the file

### Updating pyproject.toml

The `pyproject.toml` file is set up to read the version from `version.txt`. You shouldn't need to manually update the version in this file.

### Building the Package

To build the package manually:

```powershell
python -m build
```

### Uploading to PyPI

To upload to PyPI manually:

```powershell
twine upload dist/*
```

## Best Practices

- Always increment the version number for any public release.
- Use semantic versioning (MAJOR.MINOR.PATCH):
  - MAJOR: Incompatible API changes
  - MINOR: Add functionality in a backwards-compatible manner
  - PATCH: Backwards-compatible bug fixes
- Test thoroughly before releasing.
- Keep the README up to date with any significant changes.

## Troubleshooting

- If the script fails, you can perform the steps manually using the commands in the "Manual Steps" section.
- Ensure you have the necessary permissions for the GitHub repository and PyPI project.
- Check that your git configuration is correct and you're able to push to the repository.

Remember to refer to this guide whenever you need to release a new version of the project. It will help ensure a consistent and smooth release process.