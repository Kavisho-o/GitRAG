'''
this is just an inspection code
will learn how to clone a repo and read files from it
using it for testing and debugging
'''

import os
import git
import shutil   # for deleting the cloned repo after testing
from paths import CLONE_DIR

REPO_URL = "https://github.com/tiangolo/fastapi.git"

def clone_repo(url: str, dest: str):

    if os.path.exists(dest):
        print(f"Removing existing {dest}...")
        shutil.rmtree(dest, onexc=lambda f, p, e: os.chmod(p, 0o777) or f(p))  

        '''
        onexc is a callback function that will be called when an error occurs during the removal process.
        # It takes three arguments: the function that raised the error (f), 
        # the path of the file that caused the error (p), and the exception object (e).
        # The callback function changes the permissions of the file to 
        # 777 (read, write, execute for everyone) and then retries the removal.
        '''

    print(f"Cloning {url}...")
    git.Repo.clone_from(url, dest, depth=1)
    print("Done.")



def find_python_files(root: str) -> list[str]:

    '''
    Walk through the directory and its subdirectories to find all .py files.
    os.walk() generates the file names in a directory tree by walking either top-down orbottom-up. 
    For each directory in the tree rooted at the directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    '''
    py_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "venv", "__pycache__", "tests", "docs")]
        for f in filenames:
            if f.endswith(".py"):
                py_files.append(os.path.join(dirpath, f))
    return py_files
if __name__ == "__main__":

    clone_repo(REPO_URL, CLONE_DIR)
    py_files = find_python_files(CLONE_DIR)
    print(f"\nFound {len(py_files)} Python files. These are the first 10:")
    for f in py_files[:10]:
        print(f"  {f}")

    # peek at one file to know what we're chunking
    # print(f"\nSample File Content: {py_files[5]}")
    # with open(py_files[5], "r", encoding="utf-8") as fh:
    #     print(fh.read()[:800])  # printing first 800 chars