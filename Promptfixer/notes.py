import os

# Output file
output_file = "all_files_content.txt"

# Folders and files to exclude
exclude_folders = ["node_modules", ".git",".vscode","dev-dist","data"]  # Enter folder names here
exclude_files = ["package-lock.json", "all_files_content.txt"]  # Enter file names here

# Get current directory
current_dir = os.getcwd()

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk(current_dir):
        # Remove excluded folders from traversal
        dirs[:] = [d for d in dirs if d not in exclude_folders]

        for file in files:
            # Skip files in the exclusion list
            if file in exclude_files:
                continue

            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, current_dir)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                content = f"<Could not read file: {e}>"
            
            outfile.write(f"{relative_path} :\n{content}\n\n{'-'*50}\n\n")

print(f"All files (except excluded folders and files) saved in {output_file}")
