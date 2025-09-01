'''
@contact:xind2023@mail.ustc.edu.cn
@time:2025/9/1
'''


import collections
import io
import os
import random
import shutil
import sys
import time
from datetime import timedelta, datetime
from io import open
from itertools import chain

import numpy as np
import pandas as pd

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

import glob
import json
import os
import re
import operator

FORMAT_DATE = '%Y%m%d'
FORMAT_DATE_2 = '%Y-%m-%d'
FORMAT_DATE_3 = '%Y%m%d%H%M%S'


def traverse_dir_files(root_dir, ext=None, is_sorted=True):

    names_list = []
    paths_list = []
    for parent, _, fileNames in os.walk(root_dir):
        for name in fileNames:
            if name.startswith('.'):
                continue
            if ext:
                if name.endswith(tuple(ext)):
                    names_list.append(name)
                    paths_list.append(os.path.join(parent, name))
            else:
                names_list.append(name)
                paths_list.append(os.path.join(parent, name))
    if not names_list:
        return paths_list, names_list
    if is_sorted:
        paths_list, names_list = sort_two_list(paths_list, names_list)
    return paths_list, names_list


def check_np_empty(data_np):

    none_type = type(None)
    if isinstance(data_np, np.ndarray):
        if data_np.size == 0:
            return True
    elif isinstance(data_np, none_type):
        return True
    elif isinstance(data_np, list):
        if not data_np:
            return True
    else:
        return False


def sort_two_list(list1, list2, reverse=False):

    try:
        list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2), reverse=reverse)))
    except Exception as e:
        sorted_id = sorted(range(len(list1)), key=lambda k: list1[k], reverse=True)
        list1 = [list1[i] for i in sorted_id]
        list2 = [list2[i] for i in sorted_id]

    return list1, list2


def sort_three_list(list1, list2, list3, reverse=False):

    list1, list2, list3 = (list(t) for t in zip(*sorted(zip(list1, list2, list3), reverse=reverse)))
    return list1, list2, list3


def mkdir_if_not_exist(dir_name, is_delete=False):

    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return True
    except Exception as e:
        return False


def datetime_to_str(date, date_format=FORMAT_DATE):
    return date.strftime(date_format)


def str_to_datetime(date_str, date_format=FORMAT_DATE):
    date = time.strptime(date_str, date_format)
    return datetime(*date[:6])


def get_next_half_year():

    n_days = datetime.now() - timedelta(days=178)
    return n_days.strftime('%Y-%m-%d')


def timestr_2_timestamp(time_str):

    return int(time.mktime(datetime.strptime(time_str, "%Y-%m-%d").timetuple()) * 1000)


def create_folder(atp_out_dir):

    if os.path.exists(atp_out_dir):
        shutil.rmtree(atp_out_dir)
        print('文件夹 "%s" 存在，删除文件夹。' % atp_out_dir)

    if not os.path.exists(atp_out_dir):
        os.makedirs(atp_out_dir)
        print('文件夹 "%s" 不存在，创建文件夹。' % atp_out_dir)


def create_empty_file(file_name):

    if os.path.exists(file_name):
        print("文件存在，删除文件：%s" % file_name)
        os.remove(file_name)
    if not os.path.exists(file_name):
        open(file_name, 'a').close()


def remove_punctuation(line):

    rule = re.compile(u"[^a-zA-Z0-9\u4e00-\u9fa5]")
    line = rule.sub('', line)
    return line


def check_punctuation(word):
    pattern = re.compile(u"[^a-zA-Z0-9\u4e00-\u9fa5]")
    if pattern.search(word):
        return True
    else:
        return False


def clean_text(text):

    if not text:
        return ''
    return re.sub(r"\s+", " ", text)


def merge_files(folder, merge_file):

    paths, _, _ = listdir_files(folder)
    with open(merge_file, 'w') as outfile:
        for file_path in paths:
            with open(file_path) as infile:
                for line in infile:
                    outfile.write(line)


def random_pick(some_list, probabilities=None):

    if not probabilities:
        probabilities = [float(1) / float(len(some_list))] * len(some_list)

    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    item = some_list[0]
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def intersection_of_lists(l1, l2):

    return list(set(l1).intersection(set(l2)))


def safe_div(x, y):

    x = float(x)
    y = float(y)
    if y == 0.0:
        return 0.0
    r = x / y
    return r


def calculate_percent(x, y):

    x = float(x)
    y = float(y)
    return safe_div(x, y) * 100


def invert_dict(d):

    return dict((v, k) for k, v in d.items())


def init_num_dict():

    return collections.defaultdict(int)


def sort_dict_by_value(dict_, reverse=True):

    return sorted(dict_.items(), key=operator.itemgetter(1), reverse=reverse)


def sort_dict_by_key(dict_, reverse=False):

    return sorted(dict_.items(), key=operator.itemgetter(0), reverse=reverse)


def get_current_time_str():

    return datetime.now().strftime('%Y%m%d%H%M%S')


def get_current_time_for_show():

    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_current_day_str():

    return datetime.now().strftime('%Y%m%d')


def remove_line_of_file(ex_line, file_name):
    ex_line = ex_line.replace('\n', '')
    lines = read_file(file_name)

    out_file = open(file_name, "w")
    for line in lines:
        line = line.replace('\n', '')  # 确认编码格式
        if line != ex_line:
            out_file.write(line + '\n')
    out_file.close()


def map_to_ordered_list(data_dict, reverse=True):

    return sorted(data_dict.items(), key=operator.itemgetter(1), reverse=reverse)


def map_to_index(data_list, all_list):

    index_dict = {l.strip(): i for i, l in enumerate(all_list)}  # 字典
    index = index_dict[data_list.strip()]
    return index


def n_lines_of_file(file_name):

    return sum(1 for line in open(file_name))


def remove_file(file_name):

    if os.path.exists(file_name):
        os.remove(file_name)


def find_sub_in_str(string, sub_str):

    return [m.start() for m in re.finditer(sub_str, string)]


def list_has_sub_str(string_list, sub_str):

    for string in string_list:
        if sub_str in string:
            return True
    return False


def remove_last_char(str_value, num):

    str_list = list(str_value)
    return "".join(str_list[:(-1 * num)])


def read_file(data_file, mode='more'):

    try:
        with open(data_file, 'r', errors='ignore') as f:
            if mode == 'one':
                output = f.read()
                return output
            elif mode == 'more':
                output = f.readlines()
                output = [o.strip() for o in output]
                return output
            else:
                return list()
    except IOError:
        return list()


def is_file_nonempty(data_file):

    data_lines = read_file(data_file)
    if len(data_lines) > 0:
        return True
    else:
        return False


def read_csv_file(data_file, num=-1):

    import pandas
    df = pandas.read_csv(data_file)
    row_list = []
    column_names = list(df.columns)
    for idx, row in df.iterrows():
        if idx == num:
            break
        row_list.append(dict(row))
        if idx != 0 and idx % 20000 == 0:
            print('[Info] idx: {}'.format(idx))
    return column_names, row_list


def read_file_utf8(data_file, mode='more', encoding='utf8'):

    try:
        with open(data_file, 'r', encoding=encoding) as f:
            if mode == 'one':
                output = f.read()
                return output
            elif mode == 'more':
                output = f.readlines()
                output = [o.strip() for o in output]
                return output
            else:
                return list()
    except IOError:
        return list()


def read_file_gb2312(data_file, mode='more'):

    try:
        with open(data_file, 'r', encoding='gb2312') as f:
            if mode == 'one':
                output = f.read()
                return output
            elif mode == 'more':
                output = f.readlines()
                output = [o.strip() for o in output]
                return output
            else:
                return list()
    except IOError:
        return list()


def read_excel_to_df(file_path):

    df = pd.read_excel(file_path, engine='openpyxl')
    return df


def find_word_position(original, word):

    u_original = original.decode('utf-8')
    u_word = word.decode('utf-8')
    start_indexes = find_sub_in_str(u_original, u_word)
    end_indexes = [x + len(u_word) - 1 for x in start_indexes]
    return zip(start_indexes, end_indexes)


def write_list_to_file(file_name, data_list, log=False):

    if file_name == "":
        return
    with io.open(file_name, "a+", encoding='utf8') as fs:
        count = 0
        for data in data_list:
            fs.write("%s\n" % data)
            count += 1
            if count % 100 == 0 and log:
                print('[Info] write: {} lines'.format(count))

    print('[Info] final write: {} lines'.format(count))


def write_line(file_name, line):

    if file_name == "":
        return
    with io.open(file_name, "a+", encoding='utf8') as fs:
        if type(line) is (tuple or list):
            fs.write("%s\n" % ", ".join(line))
        else:
            fs.write("%s\n" % line)


def show_set(data_set):

    data_list = list(data_set)
    show_string(data_list)


def show_string(obj):

    print(list_2_utf8(obj))


def list_2_utf8(obj):

    return json.dumps(obj, encoding="UTF-8", ensure_ascii=False)


def listdir_no_hidden(root_dir):

    return glob.glob(os.path.join(root_dir, '*'))


def listdir_files(root_dir, ext=None):

    names_list = []
    paths_list = []
    for parent, _, fileNames in os.walk(root_dir):
        for name in fileNames:
            if name.startswith('.'):
                continue
            if ext:
                if name.endswith(tuple(ext)):
                    names_list.append(name)
                    paths_list.append(os.path.join(parent, name))
            else:
                names_list.append(name)
                paths_list.append(os.path.join(parent, name))
    return paths_list, names_list


def time_elapsed(start, end):

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def batch(iterable, n=1):

    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def unicode_str(s):

    try:
        s = str(s, 'utf-8')
    except Exception as e:
        try:
            s = s.decode('utf8')
        except Exception as e:
            pass
        s = s
    return s


def unfold_nested_list(data_list):

    return list(chain.from_iterable(data_list))


def unicode_list(data_list):

    return [unicode_str(s) for s in data_list]


def pairwise_list(a_list):

    if len(a_list) % 2 != 0:
        raise Exception("pairwise_list error!")
    r_list = []
    for i in range(0, len(a_list) - 1, 2):
        r_list.append([a_list[i], a_list[i + 1]])
    return r_list


def list_2_numdict(a_list):

    num_dict = collections.defaultdict(int)
    for item in a_list:
        num_dict[item] += 1
    return num_dict


def shuffle_two_list(a, b):

    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a, b


def shuffle_three_list(a, b, c):

    d = list(zip(a, b, c))
    random.shuffle(d)
    a, b, c = zip(*d)
    return a, b, c


def sorted_index(myList, reverse=True):


    idx_list = sorted(range(len(myList)), key=myList.__getitem__, reverse=reverse)
    return idx_list


def download_url_img(url):

    import cv2
    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        response = requests.get(url, verify=False)
    except Exception as e:
        print(str(e))
        return False, []
    if response is not None and response.status_code == 200:
        input_image_data = response.content
        np_arr = np.asarray(bytearray(input_image_data), np.uint8).reshape(1, -1)
        parsed_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        return True, parsed_image


def download_url_txt(url, is_split=False):

    import requests

    try:
        response = requests.get(url, timeout=3)
    except Exception as e:
        print(str(e))
        return False, []

    if response is not None and response.status_code == 200:
        text_data = response.content
        if not is_split:
            return True, text_data.decode()
        else:
            text_list = text_data.decode().splitlines()
            return True, text_list
    else:
        return False, []



def save_dict_to_json(json_path, save_dict):


    json_str = json.dumps(save_dict, indent=2, ensure_ascii=False)
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_str)


def read_json(json_path):

    import json
    json_path = json_path
    try:
        with open(json_path, 'r', encoding='utf-8') as load_f:
            res = json.load(load_f)
    except Exception as e:
        print(e)
        res = {}
    return res


def save_obj(file_path, obj):

    import pickle
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(file_path):

    import pickle
    with open(file_path, 'rb') as f:
        x = pickle.load(f)
    return x


def write_list_to_excel(file_name, titles, res_list):

    import xlsxwriter

    wk = xlsxwriter.Workbook(file_name)
    ws = wk.add_worksheet()

    for i, t in enumerate(titles):
        ws.write(0, i, t)

    for n_rows, res in enumerate(res_list):
        n_rows += 1
        try:
            for idx in range(len(titles)):
                ws.write(n_rows, idx, res[idx])
        except Exception as e:
            print(e)
            continue

    wk.close()
    print('[Info] 文件保存完成: {}'.format(file_name))


def random_prob(prob):

    x = random.choices([True, False], [prob, 1-prob])
    return x[0]


def filter_list_by_idxes(data_list, idx_list):

    res_list = []
    for idx in idx_list:
        if not isinstance(idx, list):
            res_list.append(data_list[idx])
        else:
            sub_list = []
            for i in idx:
                sub_list.append(data_list[i])
            res_list.append(sub_list)
    return res_list


def check_english_str(string):

    pattern = re.compile('^[A-Za-z0-9.,:;!?()_*"\'，。 ]+$')
    if pattern.fullmatch(string):
        return True
    else:
        return False


def get_fixed_samples(a_list, num=20000):

    if num <= 0:
        return a_list
    a_n = len(a_list)
    n_piece = num // a_n + 1
    x_list = a_list * n_piece
    random.seed(47)
    random.shuffle(x_list)
    x_list = x_list[:num]
    return x_list


def split_train_and_val(data_lines, gap=20):

    print('[Info] 样本总数: {}'.format(len(data_lines)))
    random.seed(47)
    random.shuffle(data_lines)
    train_num = len(data_lines) // gap * (gap - 1)
    train_data = data_lines[:train_num]
    val_data = data_lines[train_num:]
    print('[Info] train: {}, val: {}'.format(len(train_data), len(val_data)))
    return train_data, val_data
