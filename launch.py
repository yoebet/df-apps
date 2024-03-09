import os
import random
import sys
import ipaddress
import traceback
from threading import Thread
import multiprocessing
from contextlib import redirect_stdout, redirect_stderr
import shutil
import json
import time
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import requests
from urllib.parse import urlparse
import logging
from PIL import Image
import numpy as np

from photo_maker.inference import inference as pm_inference
from photo_maker.inference import arg_config as pm_arg_config

from utils.params import Params, build_params
from utils.dirs import get_task_dir

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass


def verify_url(url):
    import socket
    from urllib.parse import urlparse
    try:
        parsed_url = urlparse(url)
        domain_name = parsed_url.netloc
        host = socket.gethostbyname_ex(domain_name)
        for ip in host[2]:
            ip_addr = ipaddress.ip_address(ip)
            if not ip_addr.is_global:
                return False
    except Exception:
        return False

    return True


def tf_log_img(writer: SummaryWriter, tag, image_path, global_step=0):
    img = Image.open(image_path)
    if not img.mode == "RGB":
        img = img.convert("RGB")
    np_image = np.asarray(img)
    writer.add_image(tag, np_image, global_step, dataformats="HWC")


def run_sync(params: Params, *, logger, result_file: str, result, log_file: str):
    if result is None:
        result = {}

    with open(log_file, 'w') as lf:
        with redirect_stdout(lf), redirect_stderr(lf):
            try:
                result['inference_start_at'] = datetime.now().isoformat()
                pm_inference(params)
                result['success'] = True
                result['output_images_path'] = params.output_images_path
            except Exception as e:
                print(str(e), file=sys.stderr)
                traceback.print_exc()
                result['success'] = False
                result['error_message'] = str(e)
            result['finished_at'] = datetime.now().isoformat()

            json.dump(result, open(result_file, 'w'), indent=2)

    if result['success']:
        logger.info(f'task {params.task_id} Finished.')
    else:
        logger.error(f'task {params.task_id} Failed: {result["error_message"]}')

    torch.cuda.empty_cache()
    return result


def launch(config, task: Params, launch_options: Params, logger=None):
    if logger is None:
        logger = logging.getLogger('launch')

    prepare_start_at = datetime.now().isoformat()

    # logger.info(pformat(task))
    # logger.info(pformat(launch_options))
    params = task.merge(launch_options)
    if torch.cuda.is_available():
        if hasattr(launch_options, 'device_index'):
            params.device = f'cuda:{launch_options.device_index}'

            free, total = torch.cuda.mem_get_info(launch_options.device_index)
            g = 1024 ** 3
            if free < 15 * g:
                logger.warning(f'{params.task_id}: device occupied')
                return {
                    'success': False,
                    'error_message': 'device occupied',
                }
        else:
            logger.warning('device_index not set')
            params.device = 'cuda'
    else:
        params.device = 'cpu'

    TASKS_DIR = config['TASKS_DIR']
    task_dir = get_task_dir(TASKS_DIR, task.task_id, task.sub_dir)
    os.makedirs(task_dir, exist_ok=True)
    params.task_dir = task_dir
    params = build_params(params, pm_arg_config)
    json.dump(vars(params), open(f'{task_dir}/params.json', 'w'), indent=2)

    result_file = os.path.join(task_dir, 'result.json')
    if os.path.exists(result_file):
        os.remove(result_file)

    result = {
        'prepare_start_at': prepare_start_at,
    }
    log_file = os.path.join(task_dir, f'log-{str(int(time.time()))}.txt')
    logger.info(f'Logging to {log_file} ...')

    res = {'success': True, }
    args = (params)
    kwargs = {'result': result, 'result_file': result_file, 'log_file': log_file, 'logger': logger}

    if params.run_mode == 'sync':
        res = run_sync(params, **kwargs)
    elif params.run_mode == 'process':
        process = multiprocessing.Process(target=run_sync, args=args, kwargs=kwargs)
        process.start()
        res['pid'] = process.pid
    else:  # thread
        thread_name = f'thread_{params.task_id}_{random.randint(1000, 9990)}'
        # res['thread_name'] = thread_name
        thread = Thread(target=run_sync, args=args, kwargs=kwargs, name=thread_name)
        thread.start()

    return res
