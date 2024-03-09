import os
import shutil
import json
from urllib.parse import urlparse
import requests


def download(resource_url, target_dir, filename, default_ext):
    if not resource_url.startswith('http'):
        raise Exception(f'must be url: {resource_url}')
    # if not verify_url(resource_url):
    #     raise Exception(f'local resource not allowed')
    resource_path = urlparse(resource_url).path
    resource_name = os.path.basename(resource_path)
    base_name, ext = os.path.splitext(resource_name)
    if filename is None:
        filename = base_name
    if ext is None:
        ext = default_ext
    elif ext == '.jfif':
        ext = '.jpg'
    if ext is not None:
        filename = f'{filename}{ext}'

    full_path = f'{target_dir}/{filename}'
    with requests.get(resource_url, stream=True) as res:
        with open(full_path, 'wb') as f:
            shutil.copyfileobj(res.raw, f)
    return full_path


def build_params(args, config='./arg_config.json'):
    with open(config, 'r') as file:
        config_dict = json.load(file)
    for key, value in config_dict.items():
        if not hasattr(args, key):
            if not isinstance(value, dict):
                setattr(args, key, value)
            elif 'default' in value:
                setattr(args, key, value['default'])
        elif isinstance(value, dict) and 'need_change_url' in value:
            url_str = getattr(args, key)
            if isinstance(url_str, str):
                local_url = download(url_str, args.task_dir,
                                     value['default_file_name'] if 'default_file_name' in value else None,
                                     value['default_ext'])
                setattr(args, key, local_url)
            elif isinstance(url_str, list):
                local_url_strs = []
                for index, url_ in enumerate(url_str):
                    local_url_strs.append(
                        download(url_, args.task_dir, f"{value['default_file_name']}_{index}", value['default_ext']))
                setattr(args, key, local_url_strs)
            if 'sys_name' in value:
                setattr(args, value['sys_name'], getattr(args, key))
        elif isinstance(value, bool):
            str_val = getattr(args, key)
            if str_val == True or str_val == '1' or str_val.lower() == 'ture':
                setattr(args, key, True)
            else:
                setattr(args, key, False)
    return args


class Params:
    def __init__(self, param):
        for key, value in param.items():
            setattr(self, key, value)

    def merge(self, other):
        self.__dict__.update(other.__dict__)
        return self
