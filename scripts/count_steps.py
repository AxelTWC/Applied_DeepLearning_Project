import os
import glob

def count_model_responses(directory):
    total_model_responses = 0
    file_counts = {}

    # Define the pattern to search for .txt files
    search_pattern = os.path.join(directory, "question_*.txt")
    txt_files = glob.glob(search_pattern)

    if not txt_files:
        print(f"No text files found in {directory}")
        return

    print(f"Found {len(txt_files)} files to process...")

    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Count occurrences of "MODEL RESPONSE" in the file
                count = content.count("----------  MODEL RESPONSE. ----------") - 2
                file_counts[os.path.basename(file_path)] = count
                total_model_responses += count
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    # Output results
    print("-" * 40)
    print(f"Total steps in all files: {total_model_responses}")
    print(f"Average steps excluding terminate step per file: {total_model_responses / (len(txt_files) - 1)}")
    print("-" * 40)
    
    # Optional: Print counts for each file (sorted by question index for better readability)
    # Extract index from filename for sorting
    def get_index(filename):
        try:
            return int(filename.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            return -1

    sorted_files = sorted(file_counts.keys(), key=get_index)
    
    # print("Breakdown by file:")
    # for filename in sorted_files:
    #     print(f"{filename}: {file_counts[filename]}")

if __name__ == "__main__":
    target_directory = "../data/adaptive-7B/triviaqa/vanilla/rag_history"
    
    # Check if directory exists
    if os.path.exists(target_directory):
        count_model_responses(target_directory)
    else:
        print(f"Directory not found: {target_directory}")

