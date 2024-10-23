Param(
    [string]$BaseDir = "C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\LLM UTILITIES\async_llm_handler"
)

# Clear console for readability
Clear-Host

# -----------------------------
# Validate Base Directory
# -----------------------------
# Check if the base directory exists
if (!(Test-Path -Path $BaseDir)) {
    Write-Host "Error: Base directory does not exist: $BaseDir"
    exit 1
} else {
    Write-Host "Base directory exists: $BaseDir"
}

# Set current location to base directory
Set-Location -Path $BaseDir

# -----------------------------
# Create 'tests' Directory
# -----------------------------
$TestsDir = Join-Path $BaseDir "tests"

# Check if 'tests' directory exists
if (!(Test-Path -Path $TestsDir)) {
    # Create 'tests' directory
    New-Item -ItemType Directory -Path $TestsDir -Force | Out-Null
    Write-Host "Created 'tests' directory: $TestsDir"
} else {
    Write-Host "'tests' directory already exists: $TestsDir"
}

# Validate 'tests' directory creation
if (!(Test-Path -Path $TestsDir)) {
    Write-Host "Error: Failed to create 'tests' directory."
    exit 1
}

# -----------------------------
# Create Package Configuration Files
# -----------------------------

# Function to create a file with content if it doesn't exist
function Create-File {
    param (
        [string]$FilePath,
        [string]$Content
    )
    if (!(Test-Path -Path $FilePath)) {
        $Content | Out-File -FilePath $FilePath -Encoding UTF8
        Write-Host "Created file: $FilePath"
    } else {
        Write-Host "File already exists: $FilePath"
    }

    # Validate file creation
    if (!(Test-Path -Path $FilePath)) {
        Write-Host "Error: Failed to create file: $FilePath"
        exit 1
    }
}

# Create 'setup.py'
$SetupPy = Join-Path $BaseDir "setup.py"
$SetupContent = @"
from setuptools import setup, find_packages

setup(
    name='async_llm_handler',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'openai',
        'anthropic',
        'aiolimiter',
        'tiktoken',
        'pydantic',
        'python-dotenv',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='An asynchronous handler for LLM API interactions.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/async_llm_handler',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
"@
Create-File -FilePath $SetupPy -Content $SetupContent

# Create 'pyproject.toml'
$PyProjectToml = Join-Path $BaseDir "pyproject.toml"
$PyProjectContent = @"
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"
"@
Create-File -FilePath $PyProjectToml -Content $PyProjectContent

# Create 'requirements.txt'
$RequirementsTxt = Join-Path $BaseDir "requirements.txt"
$RequirementsContent = @"
openai
anthropic
aiolimiter
tiktoken
pydantic
python-dotenv
"@
Create-File -FilePath $RequirementsTxt -Content $RequirementsContent

# Create 'MANIFEST.in'
$ManifestIn = Join-Path $BaseDir "MANIFEST.in"
$ManifestContent = @"
include README.md
"@
Create-File -FilePath $ManifestIn -Content $ManifestContent

# Create '.gitignore'
$GitIgnore = Join-Path $BaseDir ".gitignore"
$GitIgnoreContent = @"
__pycache__/
*.pyc
.env
"@
Create-File -FilePath $GitIgnore -Content $GitIgnoreContent

# Create 'LICENSE'
$LicenseFile = Join-Path $BaseDir "LICENSE"
$LicenseContent = @"
MIT License

Copyright (c) $(Get-Date -Format yyyy) Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
...
"@
Create-File -FilePath $LicenseFile -Content $LicenseContent

# -----------------------------
# Create Documentation
# -----------------------------

# Create 'README.md'
$ReadmeMd = Join-Path $BaseDir "README.md"
$ReadmeContent = @"
# Async LLM Handler

## Installation

```bash
pip install async_llm_handler
```

## Usage

### Basic Usage

```python
from async_llm_handler import LLMAPIHandler

handler = LLMAPIHandler()
response = await handler.process('Your prompt here', model='gpt-4o-mini')
print(response)
```

### Advanced Usage

Provide advanced usage examples with all possible parameter combinations.

## Configuration Options

- **model**: Specify the LLM model to use.
- **system_message**: Custom system message for context.
- **temperature**: Sampling temperature.

## Troubleshooting

- Ensure your API keys are set in the environment variables or in a `.env` file.
- Check network connectivity.
- Verify model compatibility.
"@
Create-File -FilePath $ReadmeMd -Content $ReadmeContent

# -----------------------------
# Create Package Initialization File
# -----------------------------

# Create '__init__.py' in base directory
$PackageInit = Join-Path $BaseDir "__init__.py"
Create-File -FilePath $PackageInit -Content ""

# -----------------------------
# Create Test Files
# -----------------------------

# Create '__init__.py' in 'tests' directory
$TestsInit = Join-Path $TestsDir "__init__.py"
Create-File -FilePath $TestsInit -Content ""

# Create 'test_llm_handler.py' in 'tests' directory
$TestFile = Join-Path $TestsDir "test_llm_handler.py"
$TestContent = @"
import os
import unittest
from unittest.mock import patch, AsyncMock
from async_llm_handler import LLMAPIHandler

class TestLLMAPIHandler(unittest.TestCase):

    def setUp(self):
        self.handler = LLMAPIHandler()

    @patch('os.getenv')
    def test_api_key_retrieval_with_dotenv_override(self, mock_getenv):
        mock_getenv.return_value = 'test_key'
        # Test API key retrieval logic here

    def test_various_parameter_combinations(self):
        # Test different parameter combinations
        pass

    def test_error_cases_and_edge_conditions(self):
        # Test error handling and edge cases
        pass

    @patch('async_llm_handler.AsyncOpenAI')
    def test_mock_api_calls(self, mock_openai):
        # Mock API calls and test responses
        pass

    def test_environment_variable_handling(self):
        # Test handling of environment variables
        pass

if __name__ == '__main__':
    unittest.main()
"@
Create-File -FilePath $TestFile -Content $TestContent

# -----------------------------
# Validation Checks
# -----------------------------
# List of expected files and directories
$ExpectedPaths = @(
    $SetupPy,
    $PyProjectToml,
    $RequirementsTxt,
    $ManifestIn,
    $GitIgnore,
    $LicenseFile,
    $ReadmeMd,
    $PackageInit,
    $TestsInit,
    $TestFile
)

$AllValid = $true

foreach ($Path in $ExpectedPaths) {
    if (!(Test-Path -Path $Path)) {
        Write-Host "Error: Expected file not found: $Path"
        $AllValid = $false
    }
}

if ($AllValid) {
    Write-Host "`nAll files and directories have been created successfully."
    exit 0
} else {
    Write-Host "`nSome files or directories were not created successfully."
    exit 1
}
