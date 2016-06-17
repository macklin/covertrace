'''

staged.arr: cached site.arr
staged.name: cached name of site
staged.state: A list to slice staged.arr into working_arr.

'''

from __future__ import division
from os.path import join, basename, exists
from itertools import izip, izip_longest
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np




class Stage(object):
    name = None
    arr = None
    state = None
    new_file_name = 'arr_modified.npz'

staged = Stage()

'''Example of state
state = ['cytoplasm']
state = ['nuclei', 'FITC']
state = ['nuclei', 'FITC', 'area']
state = [['nuclei', 'FITC', 'area'], ['cytoplasm', 'FITC', 'area']]
'''


class data_array(np.ndarray):
    '''for 3d.
    '''

    def __new__(cls, data, labels=[]):
        # Sort index and data.

        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        sort_func = lambda x: (x[1][0], x[1][1], x[1][2])  # works with enumerate
        sort_idx = [i[0] for i in sorted(enumerate(labels), key=sort_func)]
        data = data[sort_idx, :, :]
        labels = [labels[i] for i in sort_idx]

        obj = np.asarray(data).view(cls)

        if labels:
            obj.labels = labels
        return obj

    def __getitem__(self, item):
        if isinstance(item, str):
            item = [n for n, i in enumerate(self.labels) if item == i[0]]
            if len(item) == 1 and isinstance(item, list):
                item = item[0]
            else:
                item = slice(min(item), max(item), None)
        ret = super(data_array, self).__getitem__(item)

        if hasattr(self, 'labels'):
            if isinstance(item, slice):
                ret.labels = [i[1:] for i in self.labels[item]]
        return ret

    def __dir__(self):
        return list(set([i[0] for i in self.labels]))

    def __getattr__(self, key):
        if hasattr(self, 'labels'):
            if self.labels:
                if key in set([i[0] for i in self.labels]):
                    return self[key]
        else:
            return super(data_array, self).__getattr__(key)


class cell_array(data_array):
    def __new__(cls, data, labels=[]):
        # Sort index and data.

        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        sort_func = lambda x: (x[1][0], x[1][1], x[1][2])  # works with enumerate
        single_label_idx = [a for a, aa in enumerate(labels) if len(aa) == 1]
        multi_labels = [a for a in labels if len(a) == 3]
        sort_idx = [i[0] + len(single_label_idx) for i in sorted(enumerate(multi_labels), key=sort_func)]
        for num, i in enumerate(single_label_idx):
            sort_idx.insert(i, num)
        data = data[sort_idx, :, :]
        labels = [labels[i] for i in sort_idx]

        if 'prop' not in labels:
            zero_arr = np.expand_dims(np.zeros(data[0, :, :].shape), axis=0)
            data = np.concatenate([zero_arr, data], axis=0)
            labels.insert(0, ['prop'])

        obj = np.asarray(data).view(cls)

        if labels:
            obj.labels = labels
        return obj


class Bundle(object):
    def __init__(self, parent_folder, subfolders=None, conditions=[], file_name='arr.npz'):
        parent_folder = parent_folder.rstrip('/')
        if subfolders is None:
            self.sites = Sites([parent_folder, ], file_name, [conditions, ])
        else:
            folders = [join(parent_folder, i) for i in subfolders]
            self.sites = Sites(folders, file_name, conditions)
        self.staged = staged

    def propagate(self, operations):
        """Propagate operations to all the sites.
        operations: wrapper from arr_operations
        """
        if not isinstance(operations, list):
            operations = [operations, ]
        self.sites._propagate_each(operations)

    def plotter(self, operation, fig=None, ax=None):
        # FIXME: subplots shape... how to determine?
        if fig is None:
            fig, axes = plt.subplots(1, len(self.sites), figsize=(15, 5))
            plt.tight_layout(pad=0.5)
        self.sites._plotter(operation, fig, axes)

    def merge_conditions(self):
        """Merge dataframe by conditions.
        Sites name will be also changed to conditions.
        """
        set_conditions = set([i.condition for i in self.sites])
        store_name = []
        store_condition = []
        for sc in set_conditions:
            store_name.append([i.name for i in self.sites if i.condition is sc])
            store_condition.append([i.condition for i in self.sites if i.condition is sc])
        for name, cond in zip(store_name, store_condition):
            self.sites._merge_sites(sites_name=name, condition_name=cond)

    def save(self, file_name):
        pickle.dump(self, open('{0}.pkl'.format(file_name), 'w'))



class Sites(object):
    def __init__(self, folders, file_name, conditions=[]):
        for folder, condition in izip_longest(folders, conditions):
            setattr(self, basename(folder), Site(folder, file_name, condition))

    def __iter__(self):
        __num_keys = 0
        sites_name = sorted(self.__dict__.keys())
        while len(sites_name) > __num_keys:
            yield getattr(self, sites_name[__num_keys])
            __num_keys += 1

    def _propagate_each(self, operations):
        for site in self:
            site._operate(operations)

    def _delete_file(self, file_name):
        for site in self:
            site._delete_file(file_name)

    def _merge_sites(self, sites_name, file_name, condition_name=None):
        """merge dataframe.
        - sites_name: a list of sites_name. e.g. ['A0', 'A1', 'A2']
        - file_name: a string of file_name to save
        Once implemeneted, data is saved only in the first component of
        the sites_name. The rest of attributes will be removed.
        """
        site = getattr(self, sites_name[0])
        arrs = [getattr(self, s_name).arr for s_name in sites_name]
        new_arr = np.concatenate(arrs, axis=1)
        site.save(arr=new_arr)
        [delattr(self, s_name) for s_name in sites_name[1:]]

    def _plotter(self, operation, fig, axes=None):
        try:
            axes = axes.flatten()
        except:
            axes = [axes, ]
        for site, ax in izip(self, axes):
            site._plot(operation, ax)
            ax.set_title(site.name)

    def __len__(self):
        return len([num for num, i in enumerate(self)])




class BaseSite(object):
    """name: equivalent to attribute name of Sites
    """
    merged = 0

    def __init__(self, directory, file_name, condition=None):
        self.directory = directory
        self.file_name = file_name
        self.condition = condition
        self.name = basename(directory)

    @property
    def arr(self):
        if not self.name == staged.name:
            staged.name = self.name
            staged.arr = self._read_arr(join(self.directory, self.file_name))
        return staged.arr

    @staticmethod
    def _read_arr(path):
        file_obj = np.load(path)
        return data_array(file_obj['data'], file_obj['labels'].tolist())

    def working_arr(self):
        return self._extract_working_arr()

    def _extract_working_arr(self):
        if isinstance(staged.state[0], list):
            arr_list = []
            for st in staged.state:
                arr_list.append(self._retrieve_working_arr(self.arr, st))
            return arr_list
        if isinstance(staged.state[0], str):
            return self._retrieve_working_arr(self.arr, staged.state)
        else:
            return self.arr

    @staticmethod
    def _retrieve_working_arr(arr, st):
        for num, s in enumerate(st):
            if num == 0:
                ret = arr[s]
            else:
                ret = ret[s]
        return ret

    def save(self, arr=[], labels=[]):
        dic_save = {}
        if not len(arr):
            arr = self.arr
        dic_save['data'] = arr
        if not labels:
            labels = self.arr.labels
        dic_save['labels'] = labels
        np.savez_compressed(join(self.directory, staged.new_file_name), **dic_save)
        self.file_name = staged.new_file_name
        print '\r'+'{0}: file_name is updated to {1}'.format(self.name, self.file_name),

    def _delete_file(self, arr, file_name):
        if exists(join(self.directory, file_name)):
            os.remove(join(self.directory, file_name))

    def _operate(self, operations):
        for op in operations:
            op(self.working_arr())
        self.save()


class Site(BaseSite):
    def call_prop_ops(self, op, propid, *args, **kwargs):
        if op.func.__module__ == 'ops_bool':
            bool_arr = op(self.working_arr(), *args, **kwargs)
            self.arr.prop[bool_arr] = propid
        if op.func.__module__ == 'ops_plotter':
            op(self.working_arr(propid), *args, **kwargs)

    def mark_prop_nan(self, propid):
        self.arr[:, self.arr.prop == propid] = np.nan

    def _add_null_field(self, new_label):
        zero_arr = np.expand_dims(np.zeros(self.arr[0, :, :].shape), axis=0)
        new_arr = np.concatenate([self.arr, zero_arr], axis=0)
        self.save(arr=new_arr, labels=self.arr.labels.append(new_label))
        staged.name = None

    def prop_arr(self, propid):
        new_arr = self.arr[:, (self.arr.prop == propid).any(axis=1), :]
        return cell_array(new_arr, self.arr.labels)

    def translate_prop_to_arr(self, new_label):
        self._add_null_field(new_label)
        self.arr[new_label] = self.arr.prop

    @staticmethod
    def _read_arr(path):
        file_obj = np.load(path)
        return cell_array(file_obj['data'], file_obj['labels'].tolist())

    def drop_cells(self, propid, _any=False):
        '''Drop cells.
        '''
        if not _any:
            bool_ind = -(self.arr.prop == propid).all(axis=1)
        if _any:
            bool_ind = -(self.arr.prop == propid).any(axis=1)
        new_arr = self.arr[:, bool_ind, :]
        new_arr.labels = self.arr.labels
        staged.arr = new_arr

    def working_arr(self, propid=None):
        arr = self._extract_working_arr()
        if not propid:
            return arr
        if propid:
            return arr[(self.arr.prop == propid).any(axis=1), :]


class DataHolder(object):
    def __init__(self, arr, labels):
        self.arr = arr
        self.labels = labels

    def __getitem__(self, item):
        if isinstance(item, str):
            lis = [n for n, i in enumerate(self.labels) if i[0] == item]
        elif isinstance(item, tuple):
            lis = [n for n, i in enumerate(self.labels) if i[:len(item)] == item]
        if len(lis) == 1:
            return self.arr[lis[0], :, :]
        else:
            return self.arr[slice(min(lis), max(lis), None)]





if __name__ == '__main__':
    parent_folder = '/Users/kudo/gdrive/GitHub/covertrace/data/Pos005'
    sub_folders = ['Pos005', 'Pos007', 'Pos008']
    conditions = ['IL1B', 'IL1B', 'LPS']
    site = Site(parent_folder, 'arr.npz')
    # print site.prop_arr(propid=0).shape


    from itertools import product
    obj = ['nuclei', 'cytoplasm']
    ch = ['DAPI', 'YFP']
    prop = ['area', 'mean_intensity', 'min_intensity']
    labels = [i for i in product(obj, ch, prop)]
    data = np.zeros((len(labels), 10, 5)) # 10 cells, 5 frames
    data[:, :, 1:] = 10

    dh = DataHolder(data, labels)
    print dh['nuclei'].shape
    print dh['nuclei', 'DAPI'].shape
    print dh['cytoplasm', 'DAPI', 'area'].shape
    dh['cytoplasm', 'DAPI', 'area'][5, 2] = np.Inf
    print dh.data


    # site.arr.prop = np.random.random(site.arr.cytoplasm.DAPI.area.shape)<0.05
    # print site.arr[:, cellmap.all(axis=1), :].shape
    # print site.arr[:, -cellmap.all(axis=1), :].shape
    # print site.prop_arr(propid=1).shape
    # import ops_bool
    # from functools import partial
    # staged.state = ['cytoplasm', 'DAPI', 'median_intensity']
    # op = partial(ops_bool.filter_frames_by_range, LOWER=10)
    # site.call_prop_ops(op, 1)
    # print site.prop_arr(1).shape
    # print site.prop_arr(0).shape
    # # site.arr.prop[(site.arr.prop==1).any(axis=1), :] = 1
    #
    # site.drop_cells(propid=1, _any=False)
    # print site.arr.shape
    # site.drop_cells(propid=1, _any=True)
    # print site.arr.shape
    # # print site.arr.prop
    # import ops_plotter
    # func = partial(ops_plotter.plot_all, logy=True)
    # site.call_prop_ops(func, 0)
    # plt.show()
    # # print site.arr.prop
    # # import ipdb;ipdb.set_trace()
    # # print site.cell_arr(cellmap=0).shape
    # # print site.cell_arr(cellmap=1).shape
    # # bund = Bundle(parent_folder, sub_folders, conditions, file_name='arr.npz')
    # # # bund.state = ['cytoplasm', 'DAPI', 'area']
    # # bund.staged.state = ['cytoplasm', 'DAPI', 'area']
    # # print bund.sites.Pos005.working_arr.shape
    # # bund.staged.state = ['nuclei', 'DAPI']
    # # print bund.sites.Pos005.working_arr.shape
    # # bund.staged.state = [['nuclei', 'DAPI'], ['cytoplasm', 'DAPI','area']]
    # # print bund.sites.Pos005.working_arr[0].shape, bund.sites.Pos005.working_arr[1].shape
    # bund.staged.state = [None]
    # print bund.sites.Pos005.working_arr.shape
    # bund.staged.state = ['cytoplasm', 'DAPI', 'area']
    # print bund.sites.Pos005.working_arr.shape
    # bund.merge_conditions()
    # tt = Bundle(parent_folder, sub_folders)
    # plot_all(bund.sites.Pos005.working_arr)
