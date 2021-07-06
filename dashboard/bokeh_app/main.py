from os.path import dirname, join
from functions.predict_plot import predict_plot
from functions.distribution_plot import distribution_plot
from functions.table_plot import table_plot

import pickle
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs


# import data
data = pickle.load(open(join(dirname(__file__), 'models/data_heroku.pkl'), 'rb'))

tab_table = table_plot(data)
tab_predict_plot = predict_plot(data)
tab_distri_plot = distribution_plot(data)

p = Tabs(tabs=[tab_table, tab_predict_plot, tab_distri_plot])
curdoc().add_root(p)

