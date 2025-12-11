import os
import glob

def count_retrieved_documents(directory):
    total_retrieved_docs = 0
    file_counts = {}
    files_with_final_gen = 0

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
                
                # Locate the FINAL GENERATION section
                marker = ">>>>>>>>>>  FINAL GENERATION  <<<<<<<<<<"
                parts = content.split(marker)
                
                if len(parts) > 1:
                    # The section after the marker is the final generation context
                    final_gen_content = parts[-1]
                    
                    # Count "Retrieved Contexts:" in this section
                    # Each occurrence corresponds to one sub-question's retrieval result
                    count = final_gen_content.count("Retrieved Contexts:")
                    
                    file_counts[os.path.basename(file_path)] = count
                    total_retrieved_docs += count
                    files_with_final_gen += 1
                else:
                    # If no final generation section, maybe assume 0 or handle differently
                    # For now, we assume 0 retrieved docs if process didn't reach final generation
                    file_counts[os.path.basename(file_path)] = 0
                    # print(f"Warning: No FINAL GENERATION section in {os.path.basename(file_path)}")

        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    # Output results
    print("-" * 40)
    print(f"Total retrieved documents (in Final Generation): {total_retrieved_docs}")
    print(f"Files with Final Generation section: {files_with_final_gen}")
    if files_with_final_gen > 0:
        print(f"Average retrieved documents per valid file: {total_retrieved_docs / files_with_final_gen:.2f}")
    print(f"Average retrieved documents across all files: {total_retrieved_docs / len(txt_files):.2f}")
    print("-" * 40)

if __name__ == "__main__":
    # You can change this path or pass it as an argument
    target_directory = "../data/adaptive-7B/triviaqa/vanilla/rag_history"
    
    # Check if directory exists
    if os.path.exists(target_directory):
        count_retrieved_documents(target_directory)
    else:
        # Fallback to the path user used previously if above doesn't exist
        fallback_dir = "data/adaptive-0.5B-original/nq/vanilla/rag_history"
        if os.path.exists(fallback_dir):
            print(f"Target directory not found, falling back to: {fallback_dir}")
            count_retrieved_documents(fallback_dir)
        else:
            print(f"Directory not found: {target_directory}")

