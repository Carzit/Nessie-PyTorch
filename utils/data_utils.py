import os
import json
from pathlib import Path
from typing import Union, List

def get_filename(path:str):
    file_name = os.path.basename(path)
    file_name = file_name.split('.')[0]
    return file_name

def get_extensions(extension:Union[str, List[str]]=None):
    if isinstance(extension, str):
        return [extension] if extension.startswith('.') else "."+extension
    elif isinstance(extension, list):
        return [item if item.endswith('.') else item + '.' for item in extension]
    elif extension is None:
        return None
    else:
        raise TypeError("Parameter 'extension' must be a string, a list of strings or None")

def get_files_with_extensions(dir:str, extension:Union[str, List[str]]=None) -> List[str]:
    current_dir = Path(dir)
    extensions = get_extensions(extension)
    if extensions is None:
        files = [os.path.join(dir, file.name) for file in current_dir.iterdir()]
    else:
        files = [os.path.join(dir, file.name) for file in current_dir.iterdir() if file.is_file() and file.suffix in extensions]
    return files

def check_path(path:str, extension:Union[str, List[str]]=None):
    path_obj = Path(path)
    extensions = get_extensions(extension)
    if path_obj.is_file() and (extensions is None or path_obj.suffix in extensions):
        return 1 #->file
    elif path_obj.is_dir():
        return 2 #->folder
    else:
        return 0 #->unrecognized

def merge_json_files(file_list:List):
    merged_data = {'data_X': [], 'data_Y': []}

    for file_name in file_list:
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if 'data_X' in data:
                merged_data['data_X'].extend(data['data_X'])
            if 'data_Y' in data:
                merged_data['data_Y'].extend(data['data_Y'])

    return merged_data
