from collections import abc
from rich.progress import track
from langchain.vectorstores import DeepLake
from langchain.embeddings.base import Embeddings
from IPython.display import clear_output, display, Markdown
from load_and_split import _default_text_loader, _default_text_splitter
from load_and_split import *
import importlib
import subprocess
import time
import sys
import os


def set_API(name=None, platform=None, display=False):
    if display:
        sys.argv = ['dist/decode.pyc', '-d']
    else:
        sys.argv = ['dist/decode.pyc', '-n', name, f"-{platform}"]
    spec = importlib.util.spec_from_file_location('decode', "dist/decode.pyc")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

def add_secret(name: str, key: str):
    command = ['python3', 'dist/encode.pyc', '-n', name, '-a', key]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("secret already exists")
        new_command = ['python3', 'dist/encode.pyc', '-n', name, '-a', key, '-e']
        while True:
            will = input("Do you want to edit it[y/n]: ")
            if will == "y":
                result = subprocess.run(new_command, capture_output=True, text=True)
                print("Secret edited")
                print(result.stdout)
                break
            elif will == 'n':
                print("Secret unchanged")
                break

def display_output(result, delay = 0.05):
    accumulated_text = ''
    for word in result.split(' '):
        accumulated_text += f"{word} "
        display(Markdown(accumulated_text))
        time.sleep(delay)
        clear_output(wait=True)

def Search2(path: str, relative: bool = True, ignore: dict = {"dir": [], "file": [], "extension": []}):

    if ("dir" not in ignore.keys()) or ("file" not in ignore.keys()) or ("extension" not in ignore.keys()):
        raise ValueError("ignore dictionary must contain 3 keys: 'dir', 'file', 'extension")
    if not (isinstance(ignore["dir"], abc.Sequence) or isinstance(ignore["file"], abc.Sequence) or isinstance(ignore["extension"], abc.Sequence)):
        raise ValueError("The value of each key should be a Sequence")

    if os.path.isfile(path) and (path.split(".")[-1].split("/")[-1] not in ignore["extension"]) and (path.split("/") not in ignore["file"]):
        return path
    elif os.path.isdir(path) and path.split("/")[-1] not in ignore["dir"]:
        files = []
        for p in os.listdir(path):
            if os.path.isfile(os.path.join(path, p)) and (p.split(".")[-1].split("/")[-1] not in ignore["extension"]) and (p.split("/")[-1] not in ignore["file"]):
                files.append(os.path.join(path, p))
            elif os.path.isdir(os.path.join(path, p)) and p.split("/")[-1] not in ignore["dir"]:
                files.extend(Search2(os.path.join(path, p), ignore=ignore))

        if not(relative):
            cwd = os.getcwd() + os.path.sep
            files = list(map(lambda x: cwd + x, files))

        return files
    else:
        return []
    

def Search(path: str, relative: bool = True, ignore: dict = {"dir": [], "file": [], "extension": []}):

    if ("dir" not in ignore.keys()) or ("file" not in ignore.keys()) or ("extension" not in ignore.keys()):
        raise ValueError("Wrong dictionary inserted: It must have 3 keys: 'dir', 'file', 'extension")
    if not (isinstance(ignore["dir"], list) or isinstance(ignore["file"], list) or isinstance(ignore["extension"], list)):
        raise ValueError("The value of each key should be a list")

    found_files = []
    for directory, subdirectories, files in os.walk(path):
        if directory.split("/")[-1] not in ignore["dir"]:
            for file in files:
                if file.split("/")[-1] not in ignore["file"] and file.split("/")[-1].split(".")[-1] not in ignore["extension"]:
                    found_files.append(os.path.join(directory, file))

    if not(relative):
            pwd = os.getcwd() + os.path.sep
            files = list(map(lambda x: pwd + x, files))
    return found_files
    
def output_on_file(text:str, file:str):

    with open(file, "w") as f:
        f.write(text)
        f.close()



def list_files(startpath):
    content = ""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        content = content + indent + os.path.basename(root) + "/" + "\n"
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            content = content + subindent + f + "\n"
    return content

def dir_structure(path_to_dir, ignore_files):
    files = Search2(path=path_to_dir,ignore=ignore_files)
    tree = list_files(path_to_dir)
    content = f"""Here is the list global paths of all the file in the directory:
                {files}\n\n\nAnd here is the Tree structure of the folder, Use it when asked for relative path or to print tree structure of the folder\n\n\n
    ```bash
    {tree}
    ```
    """
    with open(f"{path_to_dir}/.folder_structure.md", "w") as f:
        f.write(content)
    return f"{path_to_dir}/.folder_structure.md"




def Create_DB(files: list, database_path: str, embeddings: Embeddings, loaders: dict = {}, splitters: dict = {}):
    if not(loaders):
        loaders = Default_File_Loaders
    if not(splitters):
        splitters = Default_File_Splitters
    loaded_docs = {}
    for file in track(files, description="loading files..."):
        type_of_file = file.split("/")[-1].split(".")[-1]
        try:
            if type_of_file in loaders.keys():
                if type_of_file in loaded_docs.keys():
                    loaded_docs[type_of_file].extend(Default_File_Loaders[type_of_file](file))
                else:
                    loaded_docs[type_of_file] = Default_File_Loaders[type_of_file](file)
            else:
                if type_of_file in loaded_docs.keys():
                    loaded_docs[type_of_file].extend(_default_text_loader(file))
                else:
                    loaded_docs[type_of_file] = _default_text_loader(file)
        except Exception as e:
            print(f"[WARNING]: Could not load {file}\n")
            print(e)
    content = []
    for docs_type, docs in track(loaded_docs.items(), description="splitting files..."):
        if docs_type in splitters.keys():
            content.extend(splitters[docs_type].split_documents(docs))
        else:
            content.extend(_default_text_splitter.split_documents(docs))
    db = DeepLake(dataset_path=database_path, embedding_function=embeddings)
    db.add_documents(content)
    return db

def summarize_and_write(path_to_dir: str, ignore_files: dict, llm: str):
    summary_to_save = """This file records the contents of all other files in the given directory \
Use this file to answer the question where the user asks the name of the file which contains specific functions or values."""
    # model = ChatGroq( model_name=llm)
    # prompt = ChatPromptTemplate.from_template(template)
    files = Search2(path=path_to_dir,ignore=ignore_files)
    loaders = Default_File_Loaders
    splitters = Default_File_Splitters
    loaded_docs = {}
    for file in track(files, description="loading files..."):
        type_of_file = file.split("/")[-1].split(".")[-1]
        try:
            if type_of_file in loaders.keys():
                loaded_docs[file] = [type_of_file, Default_File_Loaders[type_of_file](file)]
            else:
                loaded_docs[type_of_file] = [type_of_file, _default_text_loader(file)]
        except Exception as e:
            print(f"[WARNING]: Could not load {file}\n")
            print(e)
    
    for docs_path, docs_info in track(loaded_docs.items(), description="splitting and summarizing files..."):
        content = []
        
        if docs_info[0] in splitters.keys():
            content.extend(splitters[docs_info[0]].split_documents(docs_info[1]))
        else:
            content.extend(_default_text_splitter.split_documents(docs_info[1]))
        # print(f"{docs_path.split('/')[-1]} {len(content)}")
        for i in content:
            memory_template = f"""The following  below content are from the the given file:
            [
            filename: {docs_path.split('/')[-1]}
            filepath: {docs_path}
            content: [{i.page_content}]
            ]"""
            # message = prompt.format_messages(path = docs_path,
            #                                  content = i)
            # summary = model(message)
            summary_to_save =  summary_to_save + "\n"+ memory_template + "\n"

        summary_to_save = summary_to_save + "\n\n"

    with open(f"{path_to_dir}/._folder_files_summary.txt", 'w') as f:
        f.write(summary_to_save)

    return f"{path_to_dir}/._folder_files_summary.txt"
