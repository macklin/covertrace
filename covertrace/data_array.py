'''

staged.arr: cached site.arr
staged.name: cached name of site
staged.state: A list to slice staged.arr into working_arr.

'''

from __future__ import division
from os.path import join, basename, exists
from itertools import izip, izip_longest, product
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.datatype_handling import sort_labels_and_arr


class Stage(object):
    name = None
    state = None
    dataholder = None
    _any = True
    new_file_name = 'arr_modified.npz'

staged = Stage()

'''Example of state
state = ['cytoplasm']
state = ['nuclei', 'FITC']
state = ['nuclei', 'FITC', 'area']
state = [['nuclei', 'FITC', 'area'], ['cytoplasm', 'FITC', 'area']]
'''

class Plotter(object):
    def __init__(self, slice_prop, operation):
        self.slice_prop = slice_prop
        self.operation = operation

    def plot(self):
        fig, axes = self._plotter(self.operation)
        return fig, axes

    def _plotter(self, operation, *args, **kwargs):
        fig, axes = self._make_fig_axes(len(self.slice_prop))
        for data, ax in zip(self.slice_prop, axes):
            if data['arr'].any():
                operation(data['arr'], ax)
                ax.set_title('{0}\n{1}/{2}/{3},\nprop={4}'.format(*[data['name']] + data['labels'] + [data['prop']]))
        return fig, axes

    def _make_fig_axes(self, num_axes):
        fig, axes = plt.subplots(1, num_axes, figsize=(15, 5), sharey=True)
        plt.tight_layout(pad=2, w_pad=0.5, h_pad=2.0)
        try:
            axes = axes.flatten()
        except:
            axes = [axes, ]
        return fig, axes


class Sites(object):
    def __init__(self, parent_folder, subfolders=None, conditions=[], file_name='arr.npz'):
        parent_folder = parent_folder.rstrip('/')
        if subfolders is None:
            folders, conditions = [parent_folder, ], [conditions, ]
        else:
            folders = [join(parent_folder, i) for i in subfolders]
        for folder, condition in izip_longest(folders, conditions):
            setattr(self, basename(folder), Site(folder, file_name, condition))
        self.staged = staged

    def __iter__(self):
        __num_keys = 0
        sites_name = sorted(self.__dict__.keys())
        sites_name.remove('staged')
        while len(sites_name) > __num_keys:
            yield getattr(self, sites_name[__num_keys])
            __num_keys += 1

    def propagate(self, operation, pid=None, *args, **kwargs):
        if 'ops_plotter' in operation.func.__module__:
            plotter = Plotter(self.collect(), operation)
            fig, axes = plotter.plot()
            return fig, axes
        else:
            for site in self:
                site.operate(operation, pid=pid)

    def collect(self):
        panels = [site.data.slice_prop for site in self]
        return [i for j in panels for i in j]

    def drop_prop(self, pid):
        for site in self:
            site._drop_prop(pid)

    def merge_conditions(self):
        """Merge dataframe by conditions.
        Sites name will be also changed to conditions.
        """
        set_cond = set([i.condition for i in self])
        group_by_cond = [[i.name for i in self if i.condition == sc] for sc in set_cond]
        for name_list in group_by_cond:
            self._merge_sites(sites_name=name_list)

    def _merge_sites(self, sites_name):
        """merge dataframe.
        - sites_name: a list of sites_name. e.g. ['A0', 'A1', 'A2']
        Once implemeneted, data is saved only in the first component of
        the sites_name. The rest of attributes will be removed.
        """
        site = getattr(self, sites_name[0])
        arrs = [getattr(self, s_name).data.arr for s_name in sites_name]
        new_arr = np.concatenate(arrs, axis=1)
        site.save(arr=new_arr)
        [delattr(self, s_name) for s_name in sites_name[1:]]

    def __len__(self):
        return len([num for num, i in enumerate(self)])


class Site(object):
    """name: equivalent to attribute name of Sites
    """
    merged = 0

    def __init__(self, directory, file_name, condition=None):
        self.directory = directory
        self.file_name = file_name
        self.condition = condition
        self.name = basename(directory)

    @property
    def data(self):
        if not self.name == staged.name:
            staged.name = self.name
            staged.dataholder = self._read_arr(join(self.directory, self.file_name))
        return staged.dataholder

    def _read_arr(self, path):
        file_obj = np.load(path)
        return DataHolder(file_obj['data'], file_obj['labels'].tolist(), self.name)

    def save(self, arr=[], labels=[], new_file_name=None):
        if not len(arr):
            arr = self.data.arr
        if not labels:
            labels = self.data.labels
        new_file_name = staged.new_file_name if not new_file_name else new_file_name
        dic_save = {'data': arr, 'labels': labels}
        np.savez_compressed(join(self.directory, new_file_name), **dic_save)
        self.file_name = staged.new_file_name
        staged.name = None
        print '\r'+'{0}: file_name is updated to {1}'.format(self.name, self.file_name),

    def operate(self, operation, pid, ax=None):
        if 'ops_bool' in operation.func.__module__:
            bool_arr = operation(self.data.slice_arr)
            self.data.prop[bool_arr] = pid
        self.save()

    def _drop_prop(self, pid):
        self.data.drop_cells(pid)
        self.save()

class DataHolder(object):
    '''
    >>> labels = [i for i in product(['nuc', 'cyto'], ['CFP', 'YFP'], ['x', 'y'])]
    >>> arr = np.zeros((len(labels), 10, 5))
    >>> print DataHolder(arr, labels)['nuc'].shape
    (4, 10, 5)
    >>> print DataHolder(arr, labels)['cyto', 'CFP'].shape
    (2, 10, 5)
    >>> print DataHolder(arr, labels)['nuc', 'CFP', 'x'].shape
    (10, 5)
    '''
    def __init__(self, arr, labels, name=None):
        labels, arr = sort_labels_and_arr(labels, arr)

        if not [i for i in labels if 'prop' in i]:
            zero_arr = np.expand_dims(np.zeros(arr[0, :, :].shape), axis=0)
            arr = np.concatenate([zero_arr, arr], axis=0)
            labels.insert(0, ['prop'])

        labels = [tuple(i) for i in labels]
        self.arr = arr
        self.labels = labels
        self.name = name

    @property
    def prop(self):
        '''Returns 2D slice of data, prop. '''
        return self['prop']

    def __getitem__(self, item):
        '''Enables dict-like behavior to extract 3D or 2D slice of arr.'''
        if isinstance(item, str):
            lis = [n for n, i in enumerate(self.labels) if i[0] == item]
        elif isinstance(item, tuple):
            lis = [n for n, i in enumerate(self.labels) if i[:len(item)] == item]
        if len(lis) == 1:
            return self.arr[lis[0], :, :]
        else:
            return self.arr[min(lis):max(lis)+1, :, :]

    @property
    def slice_arr(self):
        '''If staged.state is a list of lists, return a list of arr.
        If staged.state is a single list, return 2D or 3D numpy array.
        '''
        if isinstance(staged.state[0], list):
            arr_list = []
            for st in staged.state:
                arr_list.append(self.__getitem__(tuple(st)))
                return arr_list
        elif isinstance(staged.state[0], str):
            return self.__getitem__(tuple(staged.state))
        else:
            return self.arr

    @property
    def slice_prop(self):
        '''Return a list of dict containing array sliced by prop value.'''
        ret = []
        if isinstance(staged.state[0], str):
            slice_arr = [self.slice_arr, ]
            state = [staged.state, ]
        else:
            slice_arr = self.slice_arr
            state = staged.state
        prop_set = np.unique(self['prop'])
        for num, warr in enumerate(slice_arr):
            for pi in prop_set:
                ret.append(dict(arr=self.extract_prop_slice(warr, self.prop, pid=pi),
                                name=self.name, prop=int(pi), labels=state[num]))
        return ret

    @classmethod
    def extract_prop_slice(cls, arr, prop, pid=None):
        bool_ind = cls.retrieve_bool_ind(prop, pid)
        return np.take(arr, np.where(bool_ind)[0], axis=-2)

    @staticmethod
    def retrieve_bool_ind(prop, pid):
        func = np.any if staged._any else np.all
        return func(prop == pid, axis=1)

    def mark_prop_nan(self, pid):
        self.arr[:, self.prop == pid] = np.nan

    def _add_null_field(self, new_label):
        new_label = list(new_label) if isinstance(new_label, str) else new_label
        zero_arr = np.expand_dims(np.zeros(self.arr[0, :, :].shape), axis=0)
        self.arr = np.concatenate([self.arr, zero_arr], axis=0)
        self.labels.append(tuple(new_label))

    def translate_prop_to_arr(self, new_label):
        self._add_null_field(new_label)
        self.arr[new_label] = self.prop.copy()

    def drop_cells(self, pid):
        '''Drop cells.
        '''
        bool_ind = self.retrieve_bool_ind(self.prop, pid)
        self.arr = np.take(self.arr, np.where(-bool_ind)[0], axis=-2)

    def visit(self, visitor):
        '''visitor pattern.'''
        visitor(self)


if __name__ == '__main__':
    parent_folder = '/Users/kudo/gdrive/GitHub/covertrace/data/Pos005'
    sub_folders = ['Pos005', 'Pos007', 'Pos008']
    conditions = ['IL1B', 'IL1B', 'LPS']
    site = Site(parent_folder, 'arr.npz')
    # print site.prop_arr(propid=0).shape

    from itertools import product
    obj = ['nuclei', 'cytoplasm']
    ch = ['DAPI']
    prop = ['area', 'x']
    labels = [i for i in product(obj, ch, prop)]
    data = np.zeros((len(labels), 10, 5)) # 10 cells, 5 frames
    data[:, :, 1:] = 10

    dh = DataHolder(data, labels)
    dh['cytoplasm', 'DAPI', 'area'][5, 2] = np.Inf
    staged.state = [['cytoplasm', 'DAPI', 'area'], ['nuclei', 'DAPI', 'area']]
    dh.slice_arr
    # staged.state = [['cytoplasm', 'DAPI', 'area'], ['nuclei', 'DAPI']]
    # print site.data.arr.shape
    #
    # site.data._add_null_field(['test'])
    # print site.data.arr.shape
    # print site.data['test'].shape

    import ops_plotter
    from functools import partial

    parent_folder = '/Users/kudo/gdrive/GitHub/covertrace/data/'

    sites = Sites(parent_folder, sub_folders, conditions)
    import ipdb;ipdb.set_trace()
    sites.Pos005.data['prop'][5, :] = 2
    sites.Pos005.save()
    func = partial(ops_plotter.plot_all)
    sites.propagate(func)
    plt.show()
