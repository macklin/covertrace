'''

staged.df: cached site.df
staged.name: cached name of site
staged.state: A list to slice staged.df into working_df.

'''


from os.path import join, basename, exists
from itertools import izip, izip_longest
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np




class Stage(object):
    name = None
    df = None
    state = None

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
        sort_func = lambda x: (x[1][0], x[1][1])  # works with enumerate
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


class Bundle(object):
    def __init__(self, parent_folder, subfolders=None, conditions=[], file_name='df.npz'):
        parent_folder = parent_folder.rstrip('/')
        if subfolders is None:
            self.sites = Sites([parent_folder, ], file_name, [conditions, ])
        else:
            folders = [join(parent_folder, i) for i in subfolders]
            self.sites = Sites(folders, file_name, conditions)
        self.staged = staged

    def propagate(self, operations, file_name='df_modified.npz'):
        """Propagate operations to all the sites.
        operations: wrapper from df_operations
        """
        if not isinstance(operations, list):
            operations = [operations, ]
        self.sites._propagate_each(operations, file_name)

    def plotter(self, operation, fig=None, ax=None):
        # FIXME: subplots shape... how to determine?
        if fig is None:
            fig, axes = plt.subplots(1, len(self.sites), figsize=(15, 5))
            plt.tight_layout(pad=0.5)
        self.sites._plotter(operation, fig, axes)

    def merge_conditions(self, file_name='df_modified.npz'):
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
            self.sites._merge_sites(sites_name=name, file_name=file_name, condition_name=cond)

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

    def _propagate_each(self, operations, file_name):
        for site in self:
            df = site._operate(operations)
            site._save_df(df, file_name)

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
        dfs = [getattr(self, s_name).df for s_name in sites_name]
        new_df = np.concatenate(dfs, axis=1)
        site.save(file_name, df=new_df)
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
    def df(self):
        if not self.name == staged.name:
            staged.name = self.name
            staged.df = self._read_df(join(self.directory, self.file_name))
        return staged.df

    @staticmethod
    def _read_df(path):
        file_obj = np.load(path)
        return data_array(file_obj['data'], file_obj['labels'].tolist())

    @property
    def working_df(self):
        if isinstance(staged.state[0], list):
            df_list = []
            for st in staged.state:
                df_list.append(self._retrieve_working_df(self.df, st))
            return df_list
        if isinstance(staged.state[0], str):
            return self._retrieve_working_df(self.df, staged.state)
        else:
            return self.df

    @staticmethod
    def _retrieve_working_df(df, st):
        for num, s in enumerate(st):
            if num == 0:
                ret = df[s]
            else:
                ret = ret[s]
        return ret


    def save(self, file_name, df=[]):
        dic_save = {}
        if not len(df):
            df = self.df
        dic_save['data'] = df
        dic_save['labels'] = self.df.labels
        np.savez_compressed(join(self.directory, file_name), **dic_save)
        self.file_name = file_name
        print '\r'+'{0}: file_name is updated to {1}'.format(self.name, self.file_name),

    def _delete_file(self, df, file_name):
        if exists(join(self.directory, file_name)):
            os.remove(join(self.directory, file_name))




if __name__ == '__main__':
    parent_folder = '/Users/kudo/gdrive/GitHub/covertrace/data'
    sub_folders = ['Pos005', 'Pos007', 'Pos008']
    conditions = ['IL1B', 'IL1B', 'LPS']
    aa = Site(parent_folder, 'df.npz')
    bund = Bundle(parent_folder,sub_folders, conditions, file_name='df.npz')
    # bund.state = ['cytoplasm', 'DAPI', 'area']
    bund.staged.state = ['cytoplasm', 'DAPI', 'area']
    print bund.sites.Pos005.working_df.shape
    bund.staged.state = ['nuclei', 'DAPI']
    print bund.sites.Pos005.working_df.shape
    bund.staged.state = [['nuclei', 'DAPI'], ['cytoplasm', 'DAPI','area']]
    print bund.sites.Pos005.working_df[0].shape, bund.sites.Pos005.working_df[1].shape
    bund.staged.state = [None]
    print bund.sites.Pos005.working_df.shape


    # bund.merge_conditions()
    # tt = Bundle(parent_folder, sub_folders)
    import ipdb;ipdb.set_trace()
