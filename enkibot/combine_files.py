# enkibot/combine_files.py
# EnkiBot: Advanced Multilingual Telegram AI Assistant
# Copyright (C) 2025 Yael Demedetskaya <yaelkroy@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
import os
import logging

# --- Configuration ---
# The root directory of your project source code.
# IMPORTANT: Please ensure this path is correct for your system.
PROJECT_ROOT = r'c:\Projects\EnkiBot\EnkiBot\EnkiBot'
# The name of the file that will contain all the combined code.
OUTPUT_FILENAME = 'combined_enkibot_python_source.txt' # Changed name to reflect content
# --- End Configuration ---

# Setup basic logging for the script itself.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_all_python_files(root_dir):
    """
    Walks through the directory structure and collects paths of .py files.
    
    Args:
        root_dir (str): The starting directory to walk.
        
    Returns:
        list: A sorted list of all .py file paths found.
    """
    file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        # Exclude __pycache__ directories
        if '__pycache__' in dirpath:
            continue
        for filename in filenames:
            # --- MODIFICATION IS HERE ---
            # Only include files that end with the .py extension.
            if filename.endswith(('.py','.json')):
                file_paths.append(os.path.join(dirpath, filename))
                
    return sorted(file_paths)

def combine_project_files(root_dir, output_file):
    """
    Reads all .py files from a project directory and writes their contents
    into a single output file, with headers for each file.
    
    Args:
        root_dir (str): The root directory of the project to combine.
        output_file (str): The path to the single output file.
    """
    logging.info(f"Starting to combine only .py files from project root: '{root_dir}'")
    
    all_files = get_all_python_files(root_dir)
    
    if not all_files:
        logging.error(f"No .py files found in '{root_dir}'. Please check the PROJECT_ROOT path.")
        return

    try:
        # Open the output file in write mode with UTF-8 encoding
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for file_path in all_files:
                # Normalize path for consistent display
                normalized_path = file_path.replace('\\', '/')
                header = f"======================================================================\n"
                header += f"--- File: {normalized_path} ---\n"
                header += f"======================================================================\n\n"
                
                outfile.write(header)
                logging.info(f"Processing: {normalized_path}")
                
                try:
                    # Open the source file in read mode with UTF-8 encoding
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                    # Add newlines for separation between files
                    outfile.write("\n\n\n")
                except Exception as e:
                    # Handle potential read errors
                    error_message = f"*** ERROR: Could not read file. Reason: {e} ***\n\n\n"
                    outfile.write(error_message)
                    logging.warning(f"Could not read {file_path}: {e}")

    except IOError as e:
        logging.error(f"Fatal error writing to output file '{output_file}': {e}")
        return

    logging.info(f"Successfully combined {len(all_files)} .py files into '{output_file}'.")

if __name__ == '__main__':
    if not os.path.isdir(PROJECT_ROOT):
        print(f"Error: The project directory '{PROJECT_ROOT}' was not found.")
        print("Please make sure the PROJECT_ROOT path is correct and you are running this script from a location that can access it.")
    else:
        combine_project_files(PROJECT_ROOT, OUTPUT_FILENAME)