import os
import datetime
from pathlib import Path
import pyperclip
import requests


# =======================
# Configuration Settings
# =======================


# Exclude lists
FOLDER_EXCLUDE = {".git", "__pycache__", "node_modules", ".venv", "archive", "deployment_scripts"}
FILE_EXTENSION_EXCLUDE = {".exe", ".dll", ".so", ".pyc"}


# Define customizable tags with optional Google Docs URLs (the content of the tag will be fetched from the URL)
CUSTOM_TAGS = [
    {
        "name": "instructions",
        "url": None
    },
    {
        "name": "Error",
        "url": None  # No URL provided, will use a placeholder
    },
    # Add more tags as needed
    {
        "name": "notes",
        "url": "https://docs.google.com/document/d/1emAdwa-92zF8Jjx53qkMJX526rzIJ5ZSlxy0H2nrrzg/export?format=txt"
    },
]


# ================================
# Optional Specific Files Feature
# ================================


# Toggle to enable/disable specific files mode
INCLUDE_SPECIFIC_FILES_MODE = False # Set to True to enable specific files mode


# List of specific file paths to include (relative to root_folder)
SPECIFIC_FILES = [
    "analyze_papers.py",
    "llm_api_handler.py",
    "models.py",
    "synthesize_results.py",
    "searchers/search_result.py", # Based on the 'SearchResult' object having no 'year' attribute
]


# =========================
# Helper Function Modifications
# =========================


def obfuscate_env_value(value):
    return "********"


def create_file_element(file_path, root_folder, specific_files_set):
    relative_path = os.path.relpath(file_path, root_folder)
    file_name = os.path.basename(file_path)
    file_extension = os.path.splitext(file_name)[1]


    file_elem = f"    <file>\n"
    file_elem += f"        <name>{file_name}</name>\n"
    file_elem += f"        <path>{relative_path}</path>\n"


    # Determine if the file's content should be included
    include_content = False
    if INCLUDE_SPECIFIC_FILES_MODE:
        if relative_path in specific_files_set:
            include_content = True
    else:
        if file_extension not in FILE_EXTENSION_EXCLUDE and file_name != os.path.basename(__file__):
            include_content = True


    if include_content:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                if file_name == ".env":
                    lines = content.split("\n")
                    obfuscated_lines = []
                    for line in lines:
                        if "=" in line:
                            key, _ = line.split("=", 1)
                            obfuscated_lines.append(f"{key.strip()}={obfuscate_env_value('')}")
                        else:
                            obfuscated_lines.append(line)
                    content = "\n".join(obfuscated_lines)
                file_elem += f"        <content><![CDATA[{content}]]></content>\n"
        except UnicodeDecodeError:
            file_elem += "        <content>Binary or non-UTF-8 content not displayed</content>\n"
        except Exception as e:
            file_elem += f"        <content>Error reading file: {e}</content>\n"
    else:
        if INCLUDE_SPECIFIC_FILES_MODE:
            file_elem += "        <content>Content excluded in specific files mode</content>\n"
        else:
            file_elem += "        <content>File excluded based on extension or is the current script</content>\n"


    file_elem += "    </file>\n"
    return file_elem


def get_repo_structure(root_folder, specific_files_set):
    repo_struct = "    <repository_structure>\n"
   
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_struct += f"        <timestamp>{timestamp}</timestamp>\n"


    for root, dirs, files in os.walk(root_folder):
        # Exclude directories
        dirs[:] = [d for d in dirs if d not in FOLDER_EXCLUDE]
       
        rel_path = os.path.relpath(root, root_folder)
        dir_name = os.path.basename(root) if rel_path != "." else os.path.basename(root_folder)
        repo_struct += f'        <directory name="{dir_name}">\n'
       
        for file in files:
            file_path = os.path.join(root, file)
            repo_struct += create_file_element(file_path, root_folder, specific_files_set)


        repo_struct += "        </directory>\n"


    repo_struct += "    </repository_structure>\n"
    return repo_struct


def copy_to_clipboard(text):
    pyperclip.copy(text)


def fetch_content_from_google_doc(export_url):
    """
    Fetches the content of a public Google Doc in plain text format.


    Args:
        export_url (str): The export URL of the Google Doc.


    Returns:
        str: The content of the Google Doc.


    Raises:
        Exception: If the document cannot be fetched.
    """
    try:
        response = requests.get(export_url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Failed to fetch content. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error fetching content from {export_url}: {e}")
        return f"<!-- Failed to fetch content for this section: {e} -->"


def main():
    """
    Extracts the repository context and copies it to the clipboard.


    This function performs the following tasks:
    1. Determines the root folder of the repository.
    2. Generates a timestamp for the context.
    3. Extracts the repository structure.
    4. Constructs the XML structure with custom tags, fetching content from Google Docs if URLs are provided.
    5. Copies the entire XML to the clipboard.
    """
    root_folder = os.getcwd()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


    # Convert SPECIFIC_FILES list to a set for faster lookup if specific mode is enabled
    specific_files_set = set(SPECIFIC_FILES) if INCLUDE_SPECIFIC_FILES_MODE else set()


    # Initialize XML root
    context = "<context>\n"
   
    # Add timestamp
    context += f"    <timestamp>{timestamp}</timestamp>\n"


    # Add custom tags with content fetched from Google Docs or placeholders
    for tag in CUSTOM_TAGS:
        tag_name = tag.get("name")
        tag_url = tag.get("url")
        if tag_url:
            content = fetch_content_from_google_doc(tag_url)
        else:
            if tag_name:
                content = f"<!-- Add your {tag_name} here -->"
            else:
                content = "<!-- No tag name provided -->"
        context += f"    <{tag_name}>{content}</{tag_name}>\n"


    # Add repository structure
    context += get_repo_structure(root_folder, specific_files_set)


    # Add additional information concisely
    additional_info = (
        f"This context includes the repository structure excluding directories: {', '.join(FOLDER_EXCLUDE)} "
        f"and file extensions: {', '.join(FILE_EXTENSION_EXCLUDE)}. Sensitive information in .env files has been obfuscated. "
        f"The current script ({os.path.basename(__file__)}) is not included in the context."
    )
    if INCLUDE_SPECIFIC_FILES_MODE:
        additional_info += f" Specific files mode is enabled. Only the contents of the following files are included: {', '.join(SPECIFIC_FILES)}."
    context += f"    <additional_information>{additional_info}</additional_information>\n"


    context += "</context>"


    # Copy to clipboard
    copy_to_clipboard(context)


    print("Repository context has been copied to the clipboard.")


if __name__ == "__main__":
    main()




