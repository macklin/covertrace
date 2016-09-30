# covertrace

API to handle live-cell time-series produced by covertrack. 


### Installation
```
git clone https://github.com/braysia/covertrace.git
cd covertrace
python setup.py install
```

### Examples
Open `doc/jupyter_examples/demo.ipynb` through jupyter notebook.  


```
# Sample Usage
from functools import partial
from covertrace.data_array import Sites
from covertrace import ops_plotter, ops_filter, ops_bool

sites = Sites(parent_folder='data/sample_result/', 
              subfolders=['Pos005', 'Pos006'], conditions=['IL1B', 'IL1B'])
operation = partial(ops_plotter.plot_tsplot)
sites.set_state(['cytoplasm', 'TRITC', 'mean_intensity'])
sites.iterate(operation)
```


