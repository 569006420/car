#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import datetime
import random
import tarfile

import numpy as np
import pandas as pd
from PIL import Image

import get_logger

logger = get_logger(__name__)


class Tub(object):

    def __init__(self, path, inputs=None, types=None):

        self.path = os.path.expanduser(path)
        logger.info('path_in_tub: {}'.format(self.path))
        self.meta_path = os.path.join(self.path, 'meta.json')
        self.df = None

        exists = os.path.exists(self.path)

        if exists:
            logger.info('Tub exists: {}'.format(self.path))
            if os.path.exists(self.meta_path):
                with open(self.meta_path, 'r') as f:
                    self.meta = json.load(f)
                self.current_ix = self.get_last_ix() + 1
            else:
                self.meta = {'inputs': inputs, 'types': types}
                with open(self.meta_path, 'w') as f:
                    json.dump(self.meta, f)
                self.current_ix = 0
        elif not exists and inputs:
            logger.info('Tub does NOT exist. Creating new tub...')
            os.makedirs(self.path, 0o777)
            self.meta = {'inputs': inputs, 'types': types}
            with open(self.meta_path, 'w') as f:
                json.dump(self.meta, f)
            self.current_ix = 0
            logger.info('New tub created at: {}'.format(self.path))
        else:
            msg = "The tub path you provided doesn't exist and you didnt pass any meta info (inputs & types)" + \
                  "to create a new tub. Please check your tub path or provide meta info to create a new tub."

            raise AttributeError(msg)

        self.start_time = time.time()

    def get_last_ix(self):
        index = self.get_index()
        if len(index) >= 1:
            return max(index)
        return -1

    def update_df(self):
        df = pd.DataFrame([self.get_json_record(i) for i in self.get_index(shuffled=False)])
        self.df = df

    def get_df(self):
        if self.df is None:
            self.update_df()
        return self.df

    def get_index(self, shuffled=True):
        files = next(os.walk(self.path))[2]
        record_files = [f for f in files if f[:6] == 'record']

        def get_file_ix(file_name):
            try:
                name = file_name.split('.')[0]
                num = int(name.split('_')[1])
            except:
                num = 0
            return num

        nums = [get_file_ix(f) for f in record_files]

        if shuffled:
            random.shuffle(nums)
        else:
            nums = sorted(nums)

        return nums


    def inputs(self):
        return list(self.meta['inputs'])

    def types(self):
        return list(self.meta['types'])

    def get_input_type(self, key):
        input_types = dict(zip(self.inputs, self.types))
        return input_types.get(key)

    def write_json_record(self, json_data):
        path = self.get_json_record_path(self.current_ix)
        try:
            with open(path, 'w') as fp:
                json.dump(json_data, fp)
        except TypeError:
            logger.warn('troubles with record: {}'.format(json_data))
        except FileNotFoundError:
            raise
        except:
            logger.error('Unexpected error: {}'.format(sys.exc_info()[0]))
            raise

    def get_num_records(self):
        import glob
        files = glob.glob(os.path.join(self.path, 'record_*.json'))
        return len(files)

    def make_record_paths_absolute(self, record_dict):
        d = {}
        for k, v in record_dict.items():
            if type(v) == str: #filename
                if '.' in v:
                    v = os.path.join(self.path, v)
            d[k] = v

        return d

    def check(self, fix=False):
        logger.info('Checking tub: {}'.format(self.path))
        logger.info('Found: {} records'.format(self.get_num_records()))
        problems = False
        for ix in self.get_index(shuffled=False):
            try:
                self.get_record(ix)
            except:
                problems = True
                if fix == False:
                    logger.warning('problems with record {} : {}'.format(ix, self.path))
                else:
                    logger.warning('problems with record {}, removing: {}'.format(ix, self.path))
                    self.remove_record(ix)
        if not problems:
            logger.info('No problems found.')

    def remove_record(self, ix):
        record = self.get_json_record_path(ix)
        os.unlink(record)

    def put_record(self, data):
        json_data = {}

        for key, val in data.items():
            typ = self.get_input_type(key)

            if typ in ['str', 'float', 'int', 'boolean']:
                json_data[key] = val

            elif typ is 'image':
                name = self.make_file_name(key, ext='.jpg')
                val.save(os.path.join(self.path, name))
                json_data[key]=name

            elif typ == 'image_array':
                img = Image.fromarray(np.uint8(val))
                name = self.make_file_name(key, ext='.jpg')
                img.save(os.path.join(self.path, name))
                json_data[key]=name

            else:
                msg = 'Tub does not know what to do with this type {}'.format(typ)
                raise TypeError(msg)

        self.write_json_record(json_data)
        self.current_ix += 1
        return self.current_ix

    def get_json_record_path(self, ix):
        #return os.path.join(self.path, 'record_'+str(ix).zfill(6)+'.json')  #fill zeros
        return os.path.join(self.path, 'record_' + str(ix) + '.json')  #don't fill zeros

    def get_json_record(self, ix):
        path = self.get_json_record_path(ix)
        try:
            with open(path, 'r') as fp:
                json_data = json.load(fp)
        except UnicodeDecodeError:
            raise Exception('bad record: %d. You may want to run `python manage.py check --fix`' % ix)
        except FileNotFoundError:
            raise
        except:
            logger.error('Unexpected error: {}'.format(sys.exc_info()[0]))
            raise

        record_dict = self.make_record_paths_absolute(json_data)
        return record_dict

    def get_record(self, ix):
        json_data = self.get_json_record(ix)
        data = self.read_record(json_data)
        return data

    def read_record(self, record_dict):
        data={}
        for key, val in record_dict.items():
            typ = self.get_input_type(key)

            # load objects that were saved as separate files
            if typ == 'image_array':
                img = Image.open((val))
                val = np.array(img)

            data[key] = val
        return data

    def make_file_name(self, key, ext='.png'):
        #name = '_'.join([str(self.current_ix).zfill(6), key, ext])
        name = '_'.join([str(self.current_ix), key, ext])  # don't fill zeros
        name = name = name.replace('/', '-')
        return name

    def delete(self):
        import shutil
        shutil.rmtree(self.path)

    def shutdown(self):
        pass

    def get_record_gen(self, record_transform=None, shuffle=True, df=None):
        if df is None:
            df = self.get_df()

        while True:
            for _ in self.df.iterrows():
                if shuffle:
                    record_dict = df.sample(n=1).to_dict(orient='record')[0]

                record_dict = self.read_record(record_dict)

                if record_transform:
                    record_dict = record_transform(record_dict)

                yield record_dict

    def get_batch_gen(self, keys=None, batch_size=128, record_transform=None, shuffle=True, df=None):
        record_gen = self.get_record_gen(record_transform=record_transform, shuffle=shuffle, df=df)

        if df is None:
            df = self.get_df()

        if keys is None:
            keys = list(self.df.columns)

        while True:
            record_list = [ next(record_gen) for _ in range(batch_size) ]

            batch_arrays = {}
            for i, k in enumerate(keys):
                arr = np.array([r[k] for r in record_list])
                batch_arrays[k] = arr
            yield batch_arrays

    def get_train_gen(self, X_keys, Y_keys, batch_size=128, record_transform=None, df=None):
        batch_gen = self.get_batch_gen(X_keys + Y_keys,
                                       batch_size=batch_size,
                                       record_transform=record_transform,
                                       df=df)

        while True:
            batch = next(batch_gen)
            X = [batch[k] for k in X_keys]
            Y = [batch[k] for k in Y_keys]
            yield X, Y

    def get_train_val_gen(self, X_keys, Y_keys, batch_size=128, train_frac=.8,
                          train_record_transform=None, val_record_transform=None):
        if self.df is None:
            self.update_df()

        train_df = self.df.sample(frac=train_frac, random_state=200)
        val_df = self.df.drop(train_df.index)

        train_gen = self.get_train_gen(X_keys=X_keys, Y_keys=Y_keys, batch_size=batch_size,
                                       record_transform=train_record_transform, df=train_df)

        val_gen = self.get_train_gen(X_keys=X_keys, Y_keys=Y_keys, batch_size=batch_size,
                                     record_transform=val_record_transform, df=val_df)

        return train_gen, val_gen

    def tar_records(self, file_path, start_ix=None, end_ix=None):
        if not start_ix:
            start_ix = 0

        if not end_ix:
            end_ix = self.get_last_ix() + 1

        with tarfile.open(name=file_path, mode='w:gz') as f:
            for ix in range(start_ix, end_ix):
                record_path = self.get_json_record_path(ix)
                f.add(record_path)
            f.add(self.meta_path)

        return file_path


class TubWriter(Tub):
    def __init__(self, *args, **kwargs):
        super(TubWriter, self).__init__(*args, **kwargs)

    def run(self, *args):
        assert len(self.inputs) == len(args)
        record = dict(zip(self.inputs, args))
        self.put_record(record)


class TubReader(Tub):
    def __init__(self, *args, **kwargs):
        super(TubReader, self).__init__(*args, **kwargs)
        self.read_ix = 0

    def run(self, *args):
        if self.read_ix >= self.current_ix:
            return None

        record_dict = self.get_record(self.read_ix)
        self.read_ix += 1
        record = [record_dict[key] for key in args ]
        return record


class TubHandler():
    def __init__(self, path):
        self.path = os.path.expanduser(path)

    def get_tub_list(self):
        folders = next(os.walk(self.path))[1]
        return folders

    def next_tub_number(self):
        def get_tub_num(tub_name):
            try:
                num = int(tub_name.split('_')[1])
            except:
                num = 0
            return num

        folders = self.get_tub_list()
        numbers = [get_tub_num(x) for x in folders]
        next_number = max(numbers+[0]) + 1
        return next_number

    def create_tub_path(self):
        tub_num = self.next_tub_number()
        date = datetime.datetime.now().strftime('%y-%m-%d')
        name = '_'.join(['tub', str(tub_num).zfill(2), date])
        tub_path = os.path.join(self.path, name)
        return tub_path

    def new_tub_writer(self, inputs, types):
        tub_path = self.create_tub_path()
        tw = TubWriter(path=tub_path, inputs=inputs, types=types)
        return tw


class TubImageStacker(Tub):

    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def stack3Images(self, img_a, img_b, img_c):
        width, height, _ = img_a.shape

        gray_a = self.rgb2gray(img_a)
        gray_b = self.rgb2gray(img_b)
        gray_c = self.rgb2gray(img_c)

        img_arr = np.zeros([width, height, 3], dtype=np.dtype('B'))

        img_arr[...,0] = np.reshape(gray_a, (width, height))
        img_arr[...,1] = np.reshape(gray_b, (width, height))
        img_arr[...,2] = np.reshape(gray_c, (width, height))

        return img_arr

    def get_record(self, ix):
        data = super(TubImageStacker, self).get_record(ix)

        if ix > 1:
            data_ch1 = super(TubImageStacker, self).get_record(ix - 1)
            data_ch0 = super(TubImageStacker, self).get_record(ix - 2)

            json_data = self.get_json_record(ix)
            for key, val in json_data.items():
                typ = self.get_input_type(key)

                #load objects that were saved as separate files
                if typ == 'image':
                    val = self.stack3Images(data_ch0[key], data_ch1[key], data[key])
                    data[key] = val
                elif typ == 'image_array':
                    img = self.stack3Images(data_ch0[key], data_ch1[key], data[key])
                    val = np.array(img)

        return data



class TubTimeStacker(TubImageStacker):

    def __init__(self, frame_list, *args, **kwargs):
        super(TubTimeStacker, self).__init__(*args, **kwargs)
        self.frame_list = frame_list

    def get_record(self, ix):
        data = {}
        for i, iOffset in enumerate(self.frame_list):
            iRec = ix + iOffset

            try:
                json_data = self.get_json_record(iRec)
            except FileNotFoundError:
                pass
            except:
                pass

            for key, val in json_data.items():
                typ = self.get_input_type(key)

                # load only the first image saved as separate files
                if typ == 'image' and i == 0:
                    val = Image.open(os.path.join(self.path, val))
                    data[key] = val
                elif typ == 'image_array' and i == 0:
                    d = super(TubTimeStacker, self).get_record(ix)
                    data[key] = d[key]
                else:
                    new_key = key + "_" + str(iOffset)
                    data[new_key] = val
        return data


class TubGroup(Tub):
    def __init__(self, tub_paths_arg):
        tub_paths = files.expand_path_arg(tub_paths_arg)
        logger.info('TubGroup:tubpaths: {}'.format(tub_paths))
        self.tubs = [Tub(path) for path in tub_paths]
        self.input_types = {}

        record_count = 0
        for t in self.tubs:
            t.update_df()
            record_count += len(t.df)
            self.input_types.update(dict(zip(t.inputs, t.types)))

        logger.info('joining the tubs {} records together. This could take {} minutes.'.format(record_count,
                                                                                         int(record_count / 300000)))

        self.meta = {'inputs': list(self.input_types.keys()),
                     'types': list(self.input_types.values())}

        self.df = pd.concat([t.df for t in self.tubs], axis=0, join='inner')

    def inputs(self):
        return list(self.meta['inputs'])

    def types(self):
        return list(self.meta['types'])

    def get_num_tubs(self):
        return len(self.tubs)

    def get_num_records(self):
        return len(self.df)