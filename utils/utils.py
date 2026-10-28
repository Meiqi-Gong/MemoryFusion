import sys
import datetime
import os

def get_format_time():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

def make_log_dir(log_dir: str, name: str):
    new_log_dir = os.path.join(log_dir, f'{get_format_time()}_{name}')
    if os.path.isdir(new_log_dir):
        raise Exception(f'{new_log_dir} already exist ! abort ...')
    os.makedirs(new_log_dir)
    return new_log_dir

class Tee(object):

    def __init__(self, filename):
        self.file_name = filename
        with open(self.file_name, 'a') as f:
            pass
        self.stdout = sys.stdout

    def close(self):
        sys.stdout = self.stdout

    def write(self, data):
        with open(self.file_name, 'a') as f:
            f.write(data)
        self.stdout.write(data)

    def flush(self):
        self.stdout.flush()
