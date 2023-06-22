import subprocess
import os

def git_add_commit_push(file_path, commit_message):
    # Change to the directory containing the file
    os.chdir(os.path.dirname(file_path))
    
    # Set the GIT_TRACE environment variable to enable debug output
    os.environ['GIT_TRACE'] = '1'
    
    # Git commands
    git_add = ['git', 'add', file_path]
    git_commit = ['git', 'commit', '-m', commit_message]
    git_push = ['git', 'push', '-f', 'origin', 'main']
    
    # Execute git add, commit, and push commands
    subprocess.run(git_add, check=True)
    subprocess.run(git_commit, check=True)
    subprocess.run(git_push, check=True)
    print(f"\033[1m\033[36m\033[4m file is commited\033[0m")
path='/home/sahitya/Desktop/Adavanced_nlp/NLP_classifier'
for i,file in enumerate(os.listdir(path)):
    # Example usage
    print(f"\033[1m\033[36m\033[4m file : {i+1} are commited\033[0m")
    file_path = os.path.join(path,file)  # Path to the file you want to contribute
    commit_message = '.'  # Commit message for the contribution

    git_add_commit_push(file_path, commit_message)
    