from lime import lime_tabular

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Select, Panel
from bokeh.plotting import figure

def predict_plot(data):
    # import data (in this plot, we only need test sample, comment the else data import)
    test = data['test'].sort_index()
    train = data['train']
    #labels = data['labels']
    imputer = data['imputer']
    model = data['model']
    train_imputed = imputer.transform(train)
    test_imputed = imputer.transform(test)

    color_class = ['#718dbf', '#e84d60']
    target_class = ['non_default', 'default']

    def make_src(id):
        explainer = lime_tabular.LimeTabularExplainer(
            # subset training data to adapt local server computing power
            training_data=train_imputed[:5000, :],
            feature_names=train.columns,
            class_names=['non_default', 'default'],
            mode='classification'
        )
        client = test_imputed[test.index.get_loc(id), :]
        exp = explainer.explain_instance(
            data_row=client,
            predict_fn=model.predict_proba, num_features=6,
            # subset num_sample to adapt local serve computing power, 5000 by default
            num_samples=500
        )
        source_pred = ColumnDataSource(data={'class': target_class, 'pred': exp.predict_proba, 'color': color_class})
        var_def = []
        weight_def = []
        var_nondef = []
        weight_nondef = []
        for var_, weight_ in exp.as_map()[1]:
            if weight_ > 0:
                var_def.append(var_)
                weight_def.append(weight_)
            else:
                var_nondef.append(var_)
                weight_nondef.append(weight_)

        feature_def = train.columns[var_def].tolist()
        feature_nondef = train.columns[var_nondef].tolist()
        source_def = ColumnDataSource(dict(y=feature_def, weight=weight_def))
        source_nondef = ColumnDataSource(dict(y=feature_nondef, weight=weight_nondef))
        return source_pred, source_nondef, source_def

    # barplot for result interpretation
    def plot_exp(source_nondef, source_def):
        feature_imp = source_def.data['y'] + source_nondef.data['y']
        p_exp = figure(y_range=feature_imp, plot_height=450, plot_width=1000,
                       title='prediction explaination', tools='crosshair, pan, box_zoom, save, reset',
                       x_axis_label='contribution of features', y_axis_label='topfeatures')
        p_exp.hbar(y='y', right='weight', source=source_nondef, color=color_class[0], height=0.5,
                   legend_label=target_class[0])
        p_exp.hbar(y='y', right='weight', source=source_def, color=color_class[1], height=0.5,
                   legend_label=target_class[1])
        return p_exp

    # barplot for prediction results
    def plot_pred(exp):

        p_pred = figure(y_range=target_class, plot_height=250, plot_width=1000, x_axis_label='probability',
                        y_axis_label='credit predictions', title='prediction probability',
                        tools='hover, crosshair, pan, box_zoom, save, reset',
                        tooltips='$name: @pred')
        p_pred.hbar(source=source_pred, left=0, right='pred', y='class', color='color', height=0.5, name='score',
                    legend_field='class')
        return p_pred

    def update_plot(attrname, old, new):
        id = Select_ID.value
        # set up lime explainer
        id = int(id)
        src_pred_new, src_nondef_new, src_def_new = make_src(id)
        source_pred.data.update(src_pred_new.data)
        source_def.data.update(src_def_new.data)
        source_nondef.data.update(src_nondef_new.data)
        p2.y_range.factors = source_def.data['y'] + source_nondef.data['y']

    Select_ID = Select(title='Client_ID', value='100038', options=(test.index.astype('str').tolist()))
    Select_ID.on_change('value', update_plot)

    source_pred, source_nondef, source_def = make_src(id=100038)
    p1 = plot_pred(source_pred)
    p2 = plot_exp(source_nondef, source_def)
    p_final = Panel(child=column(Select_ID, p1, p2), title='Predict plot')
    return p_final
