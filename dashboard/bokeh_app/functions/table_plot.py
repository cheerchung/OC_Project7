
from math import floor
from os.path import dirname, join

import pickle
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, DataTable, TableColumn, NumberFormatter, \
    RangeSlider, Select, Panel


def table_plot(data):
    # import data (in this plot, we only need test sample, comment the else data import)
    test = data['test'].sort_index()

    # Set up filtering system widgets
    Select_ID = Select(title='Client_ID', value='All', options=(['All'] + test.index.astype('str').tolist()))
    Slider_AMT_Credit = RangeSlider(title="Credit Amount", start=40000, end=2170000, value=(40000, 2170000), step=1000,
                                    format="0,0")
    Select_Gender = Select(title='Gender', value='All', options=['All', 'Female', 'Male'])
    Slider_Age = RangeSlider(title='Age', start=20, end=70, value=(20, 70), step=1)

    # Set up initial source
    source_df = ColumnDataSource(data=test)

    # datatable update
    def update_info(attrname, old, new):
        # get current value
        d = Select_ID.value
        amt = Slider_AMT_Credit.value
        g = Select_Gender.value
        age = Slider_Age.value
        # generate new dataframe based on given filters
        current = (test if d == 'All' else test[test.index == int(d)])
        current = current[(current['AMT_CREDIT'] >= amt[0]) & (current['AMT_CREDIT'] <= amt[1])]
        current = (current if g == 'All' else current[current['CODE_GENDER_F'] == (1 if g == 'Female' else 0)])
        current = current[(current['DAYS_BIRTH'] > ((age[1]+1)*(-365))) & (current['DAYS_BIRTH'] <= age[0]*-365)]
        source_df.data = {
            'SK_ID_CURR': current.index,
            'AMT_CREDIT': current.AMT_CREDIT,
            'CODE_GENDER_F': current.CODE_GENDER_F.apply(lambda x: 'Female' if x == 1 else 'Male'),
            'DAYS_BIRTH': (current.DAYS_BIRTH/(-365)).apply(lambda x: floor(x))
        }

    # set columns for data table
    columns = [TableColumn(field='SK_ID_CURR', title='Client_ID'),
               TableColumn(field='AMT_CREDIT', title='Amount_Credit', formatter=NumberFormatter(format="$0,0.00")),
               TableColumn(field='CODE_GENDER_F', title='Gender'),
               TableColumn(field='DAYS_BIRTH', title='Age')]
    # set datatable
    info_table = DataTable(source=source_df, columns=columns, width=700, height=400)

    # set filtering widgets trigger
    for w in [Select_ID, Slider_AMT_Credit, Select_Gender, Slider_Age]:
        w.on_change('value', update_info),

    # Set up layouts and add to document
    inputs = column(Select_ID, Select_Gender, Slider_AMT_Credit, Slider_Age)
    # put the button and plot in a layout and add to the document
    p = Panel(child=row(inputs, info_table), title='Filter table')

    return p



