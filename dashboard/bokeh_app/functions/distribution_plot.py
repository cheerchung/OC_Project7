from scipy.stats.kde import gaussian_kde
import pandas as pd
import numpy as np
from scipy.stats.kde import gaussian_kde
import pickle
import re
from os.path import dirname, join

from bokeh.plotting import figure
from bokeh.layouts import column, row, Spacer
from bokeh.io import curdoc
from bokeh.transform import dodge
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter, LabelSet, Select, PreText, Panel


def distribution_plot(data):
    pd.set_option('max_colwidth', 200)
    color_class = ['#718dbf', '#e84d60']
    test = data['test'].sort_index()
    train = data['train']
    labels = data['labels']
    imputer = data['imputer']
    #model = data['model']
    train_imputed = imputer.transform(train)
    #test_imputed = imputer.transform(test)
    # description csv
    desp = data['description']

    def make_src(var_name):
        if train[var_name].nunique() <= 3:
            # set up source for bar distribution
            non_default = train[labels == 0][var_name].value_counts(normalize=True,
                                                                    sort=False)
            default = train[labels == 1][var_name].value_counts(normalize=True,
                                                                sort=False)
            non_default.name = 'non_default';
            default.name = 'default'
            df_bar_distribution = pd.concat([non_default, default], axis=1).fillna(0)
            source_distri = ColumnDataSource(dict(
                feature=df_bar_distribution.index.tolist(),
                non_default=df_bar_distribution['non_default'].values,
                default=df_bar_distribution['default'].values
            ))
        else:
            # set up source for kde distribution for continuous variable
            # kernel density estimation of feature
            pdf_default = gaussian_kde(train[labels == 1][var_name].dropna())
            pdf_non_def = gaussian_kde(train[labels == 0][var_name].dropna())
            # range of x_axis
            lf, rt = train[var_name].min(), train[var_name].max()
            # x, y coordinate
            x_pdf = np.linspace(lf, rt, 100)
            y1, y0 = pdf_default(x_pdf), pdf_non_def(x_pdf)
            source_distri = ColumnDataSource(dict(x=x_pdf, y_non_def=y0, y_def=y1))

        # calculation of median
        m_non_default = train[labels == 0][var_name].median()
        m_default = train[labels == 1][var_name].median()
        # median coordinate
        x_med = [m_non_default, m_default]
        pdf_default = gaussian_kde(train[labels == 1][var_name].dropna())
        pdf_non_def = gaussian_kde(train[labels == 0][var_name].dropna())
        y_med = [pdf_non_def(m_non_default)[0], pdf_default(m_default)[0]]

        # show median in graph
        source_med = ColumnDataSource(data=dict(x_med=x_med, y_med=y_med, color=color_class))

        return source_distri, source_med


    def distribution_plot(source_distri, source_med, var_name):
        # if it is categorical feature (n_unique<=3), use bar plot:
        if train[var_name].nunique() <= 3:
            # barplot for categorical feature
            p = figure(plot_width=600, plot_height=400,  # x_range=source_distri.data[var_name],
                       title=var_name + ' Distribution',
                       x_axis_label=var_name, y_axis_label='ratio',
                       tools='hover, crosshair, pan, box_zoom, save, reset',
                       tooltips='$name: @$name{%0.0}<br>')
            # barplot for non default loans
            p.vbar(x=dodge('feature', -0.125), top='non_default', width=0.2,
                   source=source_distri, color=color_class[0], name='non_default',
                   legend_label='non_default')
            # barplot for default loans
            p.vbar(x=dodge('feature', 0.125), top='default', width=0.2,
                   source=source_distri, color=color_class[1], name='default',
                   legend_label='default')
            p.yaxis.formatter = NumeralTickFormatter(format='0.0%')
            p.xaxis.ticker = source_distri.data['feature']

        else:
            # kde plot for numeric feature
            hover = HoverTool(names=['Median'])
            p = figure(plot_width=600, plot_height=400, x_axis_label=var_name,
                       y_axis_label='density', title=var_name + ' Distribution',
                       tools=[hover, 'crosshair, pan, box_zoom, save, reset'],
                       tooltips='$name: @x_med')
            # kde plot for default loans
            p.line('x', 'y_def', color=color_class[1], name='default',
                   line_width=2, legend_label='default', source=source_distri)
            # kde plot for non default loans
            p.line('x', 'y_non_def', color=color_class[0], name='non_default',
                   line_width=2, legend_label='non_default', source=source_distri)

            p.circle(x='x_med', y='y_med', size=8, color='color', line_width=2,
                     fill_alpha=0.2, name='Median', source=source_med)

        return p


    source_distri_line, source_med_line = make_src('EXT_SOURCE_3')
    p_line = distribution_plot(source_distri_line, source_med_line, 'EXT_SOURCE_3')
    source_distri_bar, source_med_bar = make_src('REGION_RATING_CLIENT')
    p_bar = distribution_plot(source_distri_bar, source_med_bar, 'REGION_RATING_CLIENT')


    def update_plot_line(attrname, old, new):
        feature = Select_feature_numeric.value
        feature = str(feature)
        src_distri_new, src_med_new = make_src(feature)
        source_distri_line.data.update(src_distri_new.data)
        source_med_line.data.update(src_med_new.data)
        p_line.title.text = '%s Distribution'%feature
        p_line.xaxis.axis_label = feature


    def update_plot_bar(attrname, old, new):
        feature = Select_feature_category.value
        src_distri_new, _ = make_src(feature)
        source_distri_bar.data.update(src_distri_new.data)
        p_bar.xaxis.ticker = source_distri_bar.data['feature']
        p_bar.title.text = '%s Distribution'%feature
        p_bar.xaxis.axis_label = feature


    def feature_description(desp, var_name):
        '''This function gives description of features'''
        cols = []
        for col_ in desp['Row']:
            if bool(re.search(col_, var_name)):
                cols.append(col_)
        # the maximum subset is most likely to be the var_name
        col = max(cols, key=len)
        df = desp[desp['Row'] == col][['Row', 'Description', 'Table']]
        text = ''
        for x, y, z in zip(df['Row'], df['Table'], df['Description']):
            text += '\"{}\"in table \"{}\" means:\n{}.\n\n'.format(x, y, z)
        return text


    def update_info(attrname, old, new):
        feature_cat = Select_feature_category.value
        feature_num = Select_feature_numeric.value
        id_cat = int(Select_ID_cat.value)
        id_num = int(Select_ID_num.value)
        info_cat = test[test.index == id_cat][[feature_cat]]
        desp_cat = feature_description(desp, feature_cat)
        desp_num = feature_description(desp, feature_num)
        info_num = test[test.index == id_num][[feature_num]]
        info_cat.index.name = 'Client_ID'
        info_num.index.name = 'Client_ID'
        client_info_cat.text = 'client info:\n' + str(info_cat) + '\n\n\n'+'feature info:\n' + str(desp_cat)
        client_info_num.text = 'client info:\n' + str(info_num) + '\n\n\n'+'feature info:\n' + str(desp_num)


    client_info_cat = PreText(text='client_info:', width=500)
    client_info_num = PreText(text='client_info:', width=500)


    col_category = train.nunique()[train.nunique().values <= 3].sort_index().index.tolist()
    col_numeric = train.nunique()[train.nunique().values > 3].sort_index().index.tolist()
    Select_feature_category = Select(title='feature_category', value='REGION_RATING_CLIENT', options=col_category)
    Select_feature_numeric = Select(title='feature_numeric', value='EXT_SOURCE_3', options=col_numeric)
    Select_ID_cat = Select(title='Client_ID', value='100038', options=test.index.astype('str').tolist())
    Select_ID_num = Select(title='Client_ID', value='100038', options=test.index.astype('str').tolist())

    Select_feature_numeric.on_change('value', update_plot_line)
    Select_feature_category.on_change('value', update_plot_bar)
    for s in [Select_ID_cat, Select_ID_num, Select_feature_numeric, Select_feature_category]:
        s.on_change('value', update_info)

    graph_cat = column(Select_feature_category, p_bar)
    graph_numeric = column(Select_feature_numeric, p_line)
    info_cat = column(Select_ID_cat, client_info_cat)
    info_numeric = column(Select_ID_num, client_info_num)
    up = row(graph_cat, Spacer(width=50), info_cat, )
    down = row(graph_numeric, Spacer(width=50), info_numeric)
    p_final = Panel(child=column(up, down ), title='Compare plot')
    return p_final