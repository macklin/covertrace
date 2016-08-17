'''


Due to the memory issue, we do not want to load all the data at once.


'''

from __future__ import division
from os.path import join, basename, exists
from itertools import izip, izip_longest, product
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.datatype_handling import sort_labels_and_arr
import re
from scipy.ndimage import imread
from functools import partial
from image_vis import ImageVis


class Stage(object):
    name = None
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

    def set_state(self, state):
        for site in self:
            site._set_state(state)

    def __iter__(self):
        __num_keys = 0
        sites_name = sorted(self.__dict__.keys())
        sites_name.remove('staged')
        while len(sites_name) > __num_keys:
            yield getattr(self, sites_name[__num_keys])
            __num_keys += 1

    def iterate(self, operation, pid=None, *args, **kwargs):
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

    def propagate_prop(self, pid):
        for site in self:
            site._propagte_prop(pid)

    def drop_prop(self, pid):
        for site in self:
            site._drop_prop(pid)

    def reset_prop(self):
        for site in self:
            site._reset_prop()

    def merge_conditions(self):
        """Merge sites if they have the same conditions.
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
    _state = None

    def __init__(self, directory, file_name, condition=None):
        self.directory = directory
        self.file_name = file_name
        self.condition = condition
        self.name = basename(directory)
        self._staged = staged

    def _set_state(self, state):
        self._state = state
        self.data._state = state

    @property
    def data(self):
        if not self.name == staged.name:
            staged.name = self.name
            staged.dataholder = self._read_arr(join(self.directory, self.file_name))
        return staged.dataholder

    def _read_arr(self, path):
        file_obj = np.load(path)
        return DataHolder(file_obj['data'], file_obj['labels'].tolist(), self.name, self._state)

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

    def operate(self, operation, pid=1, ax=None):
        if 'ops_bool' in operation.func.__module__:
            # assign pid to cells based on bool_arr returned by operation
            bool_arr = operation(self.data.slice_arr)
            self.data.prop[bool_arr] = pid
        if 'ops_filter' in operation.func.__module__:
            # Does not explicitly change the array but it is modified inside.
            operation(self.data.slice_arr)
        if 'ops_sort' in operation.func.__module__:
            #NEW
            sort_idx = operation(self.data.slice_arr)
            self.data.arr[:] = self.data.arr[sort_idx, :, :]
        self.save()

    def _drop_prop(self, pid):
        self.data.drop_cells(pid)
        self.save()

    def _reset_prop(self):
        self.data['prop'][:] = np.zeros(self.data.prop.shape)
        self.save()

    def _propagte_prop(self, pid):
        prop = self.data['prop']
        prop[(prop == pid).any(axis=1), :] = pid
        self.save()

    @property
    def images(self):
        objects = set([i[0] for i in self.data.labels if len(i) == 3])
        channels = set([i[1] for i in self.data.labels if len(i) == 3])
        return ImageHolder(self.directory, channels, objects, self._state)


class ImageHolder(object):
    def __init__(self, directory, channels, objects, state):
        self.dir = directory
        for ch in channels:
            setattr(self, ch, partial(self._channels, ch=ch))
        for ob in objects:
            setattr(self, ob, partial(self.outlines, ob=ob))
        self._state = state
        self.visualize = ImageVis(self, staged.dataholder, self._state)

    def _retrieve_file_name_by_frame(self, subfolder, frame):
        files = os.listdir(join(self.dir, subfolder))
        refiles = [re.match('img_(?P<frame>[0-9]*)_', i) for i in files]
        return [i.string for i in refiles if int(i.group('frame')) == frame]

    def _channels(self, frame, ch, rgb=False):
        file_names = self._retrieve_file_name_by_frame('channels', frame)
        ch_file_name = [i for i in file_names if ch in i][0]
        img = imread(join(self.dir, 'channels', ch_file_name))
        if rgb:
            return np.moveaxis(np.tile(img, (3, 1, 1)), 0, 2)
        else:
            return img

    def outlines(self, frame, ob):
        file_names = self._retrieve_file_name_by_frame('outlines', frame)
        obj_file_name = [i for i in file_names if ob in i][0]
        return imread(join(self.dir, 'outlines', obj_file_name))



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
    def __init__(self, arr, labels, name=None, state=None):
        if not [i for i in labels if 'prop' in i]:
            zero_arr = np.expand_dims(np.zeros(arr[0, :, :].shape), axis=0)
            arr = np.concatenate([zero_arr, arr], axis=0)
            labels.insert(0, ['prop'])
        if isinstance(labels[0], tuple):
            labels = [list(i) for i in labels]
        labels, arr = sort_labels_and_arr(labels, arr)
        labels = [tuple(i) for i in labels]
        self.arr = arr
        self.labels = labels
        self.name = name
        self._state = state

    @property
    def prop(self):
        '''Returns 2D slice of data, prop. '''
        return self['prop']

    def __getitem__(self, item):
        '''Enables dict-like behavior to extract 3D or 2D slice of arr.'''
        if isinstance(item, str):
            lis = [n for n, i in enumerate(self.labels) if i[0] == item]
        elif isinstance(item, tuple) or isinstance(item, list):
            lis = [n for n, i in enumerate(self.labels) if tuple(i[:len(item)]) == tuple(item)]
        if len(lis) == 1:
            return self.arr[lis[0], :, :]
        else:
            return self.arr[min(lis):max(lis)+1, :, :]

    @property
    def slice_arr(self):
        '''If state is a list of lists, return a list of arr.
        If state is a single list, return 2D or 3D numpy array.
        '''
        if isinstance(self._state[0], list):
            arr_list = []
            for st in self._state:
                arr_list.append(self.__getitem__(tuple(st)))
            return arr_list
        elif isinstance(self._state[0], str):
            return self.__getitem__(tuple(self._state))
        else:
            return self.arr

    @property
    def slice_prop(self):
        '''Return a list of dict containing array sliced by prop value.'''
        ret = []
        if isinstance(self._state[0], str):
            slice_arr = [self.slice_arr, ]
            state = [self._state, ]
        else:
            slice_arr = self.slice_arr
            state = self._state
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


if __name__ == '__main__':
    pass
