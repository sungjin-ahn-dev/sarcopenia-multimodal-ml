import configparser
import pickle
import os
from shutil import copyfile
from multiprocessing import Pool
from contextlib import closing
import pandas as pd


def save_obj(path, obj):
    '''
    pickle 파일로 출력
    :param path:
    :param obj:
    :return:
    '''
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    '''
    pickle 파일 로드
    :param path:
    :return:
    '''
    with open(path, 'rb') as f:
        return pickle.load(f)


class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_dict = {'tag': []}
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            self.log_dict = df.to_dict(orient='list')

    def log(self, tag, loss_dict=None):
        log_dict = self.log_dict
        log_dict['tag'].append(tag)
        if loss_dict is not None:
            for key, value in loss_dict.items():
                if key not in log_dict:
                    log_dict[key] = []
                log_dict[key].append(value)

        df = pd.DataFrame(log_dict)
        df.to_csv(self.log_path, index=False)


def check_directoty(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ConfigManager():
    #DEFAULT_PATH = './config/config.ini'
    SECTION_DEFAULT = 'DEFAULT'
    config_path = None

    def __init__(self, file_name='./config/config.ini'):
        self.config_path = file_name

    def copy_file(self, output_path):
        copyfile(self.DEFAULT_PATH, output_path)

    def load(self, input_path=None):
        config_file = configparser.ConfigParser()
        if input_path is None:
            input_path = self.config_path

        config_file.read(input_path)
        return config_file[self.SECTION_DEFAULT]

    def save(self, config_dict, output_path=None):
        if output_path is None:
            output_path = self.config_path

        if output_path is None:
            return

        config_file = configparser.ConfigParser()

        for key in config_dict:
            val = None
            if type(config_dict[key]) == int:
                val = "%d" % config_dict[key]
            elif type(config_dict[key]) == float:
                val = "%f" % config_dict[key]
            elif type(config_dict[key]) == bool:
                if config_dict[key]:
                    val = 'True'
                else:
                    val = 'False'
            elif type(config_dict[key]) == str:
                val = config_dict[key]

            if '%s' in val:
                val = val.replace('%s', '%%s')

            if val:
                config_file[self.SECTION_DEFAULT][key] = val

        with open(output_path, 'w') as writeFile:
            config_file.write(writeFile)

    # 현재 설정 내용을 출력함
    def dump(self, config_dict):
        for key in config_dict:
            if type(config_dict[key]) == int:
                print("%s = %d[int]" % (key, config_dict[key]))
            elif type(config_dict[key]) == float:
                print("%s = %f[float]" % (key, config_dict[key]))
            elif type(config_dict[key]) == bool:
                if config_dict[key]:
                    print("%s = True[bool]" % key)
                else:
                    print("%s = False[bool]" % key)
            elif type(config_dict[key]) == str:
                print("%s = %s[str]" % (key, config_dict[key]))

class BatchProcessor:
    PCS_MAX = 8
    def __init__(self, fnc, use_parallel, proc_num=None):
        self.fnc = fnc
        self.use_parallel = use_parallel
        if proc_num is None:
            proc_num = self.PCS_MAX
        self.proc_num =proc_num

    def run(self, args_lst):
        fnc, use_parallel = self.fnc, self.use_parallel
        if use_parallel:
            with closing(Pool(self.proc_num)) as p:
                result_lst = p.starmap(fnc, args_lst)
        else:
            result_lst = []
            for args in args_lst:
                result = fnc(*args)
                if result is not None:
                    result_lst.append(result)
        return result_lst
