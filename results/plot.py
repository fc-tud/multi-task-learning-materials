#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

ROOT_DIR = os.getcwd()
METRIC_LIST = ['rmse', 'mse', 'mae', 'r2', 'mape']
SIZE_DICT = {'uci': {'size': 1030, 'feat': 8}, 'uci-cbm': {'size': 11934, 'feat': 16},
             'atici': {'size': 140, 'feat': 3}, 'bachir': {'size': 112, 'feat': 3}, 'koya': {'size': 110, 'feat': 10},
             'huang': {'size': 114, 'feat': 9}, 'hu_tensile-strength': {'size': 896, 'feat': 27},
             'hu_yield-strength': {'size': 860, 'feat': 27}, 'hu_elongation': {'size': 783, 'feat': 27},
             'yin': {'size': 900, 'feat': 11}, 'su_bond-1': {'size': 122, 'feat': 7},
             'su_bond-2': {'size': 136, 'feat': 5}, 'xiong': {'size': 43, 'feat': 4},
             'guo': {'size': 63162, 'feat': 27}, 'guo-reduced-1000': {'size': 1000, 'feat': 27},
             'mat-bench': {'size': 312, 'feat': 14}, 'drop-tower': {}}
LIT_DICT = {'koya_rup': {'range': [0.238, 1.26]},
            'koya_compressive': {'range': [0.568, 1.23]},
            'hu_tensile-strength': {'range': [0.9901, 1.0099]},
            'hu_yield-strength': {'range': [0.9824, 1.0176]},
            'hu_elongation': {'range': [0.9417, 1.0583]}
            }


dict_mode = {'STL':                     {'colour':'#2E2960'},
             'MTL-predict-all':         {'colour':'#61c9cf'},
             'MTL-predict-other':       {'colour':'#5481C5'},
             'MTL-predict-all-unc':     {'colour':'#918DC3'},
             'MTL-predict-other-unc':   {'colour':'#2E2960'},
             'MTL-true-other':          {'colour':'#62cfac'},
             'FFNN_input':              {'colour':'#61c9cf'},
             'FFNN_stl':                {'colour':'#000000'}, 
             'FFNN_mtl':                {'colour':'#5481C5'},
             'MMOE':                    {'colour':'#62cfac'},
             'MTLNET':                  {'colour':'#2E2960'}
             }

dict_resources = {'15_min':                  {'colour':'#00305e'},
                  '5_min':                   {'colour':'#918DC3'},
                  '1_min':                   {'colour':'#57cca6'},
                 }


dict_mtl_mode = {'STL':  {'colour':'#57cca6', 'fill-colour':'#d6fff2'},
                 'MTL-RC':{'colour':'#00305e', 'fill-colour':'#b4cade'},
                 'MTL-NN':{'colour':'#00305e', 'fill-colour':'#b4cade'}
                }

name_mapping = {
    'MTL-true-other': 'true-other',
    'MTL-predict-other': 'pred-other',
    'MTL-predict-all': 'pred-all',
    'MTL-predict-other-unc': 'pred-other-unc',
    'MTL-predict-all-unc': 'pred-all-unc',
    'MTL-RC': 'MTL-RC-combined',
    'MTL-NN': 'MTL-NN-combined'
}

def get_results_df(model_class=['1_auto_ml', '2_pytorch']):
    # Loop trough results folder and read summary.csv
    results_dir = os.path.join(ROOT_DIR)
    df_all = pd.DataFrame()
    for model_class in model_class:
        model_class_dir = os.path.join(results_dir, model_class)
        for framework in os.listdir(model_class_dir):
            framework_dir = os.path.join(model_class_dir, framework)
            for dataset in (x for x in os.listdir(framework_dir) if x[0] != '~'):
                df = pd.read_csv(os.path.join(framework_dir, dataset, 'regression_summary.csv'))
                df['model_class'] = model_class.split("_", 1)[1]
                df['framework'] = framework
                df['dataset'] = dataset.split("_", 1)[0]
                # df['Task'] = [dataset.split("_", 1)[0] + '_' + Task for Task in df['Task']]
                df['sparse_mode'] = '_'.join(dataset.split("_")[1:2])
                df['sparse_amount'] = '_'.join(dataset.split("_")[2:3])
                df['resources'] = '_'.join(dataset.split("_")[3:5])
                df_all = pd.concat([df_all, df])

    # Add general Metadata
    df_all = df_all.reset_index(drop=True)
    df_all['mtl_mode'] = 'nan'
    df_all.loc[df_all['model_class'] == 'auto_ml', 'mtl_mode'] = \
        df_all['mode'].loc[df_all['model_class'] == 'auto_ml'].apply(lambda x: 'STL' if x == 'STL' else 'MTL-RC')
    df_all.loc[df_all['model_class'] == 'pytorch', 'mtl_mode'] = 'MTL-NN'

    # Append size and shape_ratio
    df_all['dataset-key'] = df_all['dataset']
    # Extra for Su and Hu, because of 2 different dataset sizes for Su, and Hu Task depending
    for dataset in ['su', 'hu']:
        df_all.loc[df_all['dataset'] == dataset, 'dataset-key'] = dataset + '_' + \
                                                                  df_all.loc[df_all['dataset'] == dataset, 'Task']
    df_all['size'] = df_all['dataset-key'].map(lambda x: SIZE_DICT[x]['size'])
    df_all['shape-ratio'] = df_all['dataset-key'].map(lambda x: SIZE_DICT[x]['size']/SIZE_DICT[x]['feat'])
    df_all = df_all.drop('dataset-key', axis=1)

    df_all['dataset-task-key'] = df_all['dataset'] + '_' + df_all['Task']
    df_all['dataset-task-key'] = df_all['dataset-task-key'].str.replace('guo-reduced-1000', 'guo')

    return df_all  # .sort_values(by=['size','Task'],ascending=[True, False])


def calc_relative_lit_scores(df):
    # Calc relative scores
    df['relative_score_lit'] = np.nan
    for task in df['dataset-task-key'].unique():
        for metric in METRIC_LIST:
            if metric == 'r2':
                df.loc[df['dataset-task-key'] == task, f'relative_{metric}_lit'] = \
                    df.loc[df['dataset-task-key'] == task, metric] / df_lit[metric].loc[df_lit['Task'] == task].max()
            else:
                df.loc[df['dataset-task-key'] == task, f'relative_{metric}_lit'] = \
                    (df.loc[df['dataset-task-key'] == task, metric] / df_lit[metric].loc[df_lit['Task'] == task].min())
            if df.loc[df['dataset-task-key'] == task, 'relative_score_lit'].isnull().values.any():
                # and not(df_all.loc[x,'relative_{}'.format(metric)]==0):
                df.loc[df['dataset-task-key'] == task, 'relative_score_lit'] = \
                    df.loc[df['dataset-task-key'] == task, f'relative_{metric}_lit']

    return df.sort_values(by=['size', 'Task'], ascending=[True, False])


def calc_relative_stl_scores(df, get_best_stl = f"`mode` == 'STL'  and `best-of-mode` == 1"):
    # Calc relative scores
    df_list = []
    df['relative_score_stl'] = np.nan
    if 'best-of-mode' not in df.columns:
        grouped = group_results(df, 'mode')
        get_best_results(df, grouped, modi=['mode', 'mtl_mode'], target='relative_score_lit_median', optimum='max')
    for group, group_df in df.groupby(['dataset-task-key', 'sparse_mode', 'sparse_amount', 'resources']):
        # Müssen hier die Ressourcen dazu?
        # Oder wir gehen hier über den Ressourcen df
        # 2 mal durchführen, einmal für 5 min einmal für 15 min
        
        #print(group_df.query(get_best_stl))
        for metric in METRIC_LIST:
            if metric == 'r2':
                group_df[f'relative_{metric}_stl'] = \
                    group_df[metric] / group_df.query(get_best_stl)[metric].median()
            else:
                group_df[f'relative_{metric}_stl'] = \
                    1 / (group_df[metric] / group_df.query(get_best_stl)[metric].median())
            if group_df['relative_score_stl'].isnull().values.any():
                # and not(df_all.loc[x,'relative_{}'.format(metric)]==0):
                group_df['relative_score_stl'] = group_df[f'relative_{metric}_stl']
        df_list.append(group_df)

    df = pd.concat(df_list)
    return df.sort_values(by=['size', 'Task'], ascending=[True, False])


def calc_relative_xgboost_scores(df):
    # Calc relative scores
    df_list = []
    df['relative_score_xgboost'] = np.nan
    for group, group_df in df.groupby(['dataset-task-key', 'mtl_mode', 'sparse_mode', 'sparse_amount']):
        get_xgboost = f"`framework` == 'XGBoost' and `best-of-mtl_mode` == 1"
        for metric in METRIC_LIST:
            if metric == 'r2':
                group_df[f'relative_{metric}_xgboost'] = \
                    group_df[metric] / group_df.query(get_xgboost)[metric].median()
            else:
                group_df[f'relative_{metric}_xgboost'] = \
                    1 / (group_df[metric] / group_df.query(get_xgboost)[metric].median())
            if group_df['relative_score_xgboost'].isnull().values.any():
                # and not(df_all.loc[x,'relative_{}'.format(metric)]==0):
                group_df['relative_score_xgboost'] = group_df[f'relative_{metric}_xgboost']
        df_list.append(group_df)

    df = pd.concat(df_list)
    return df.sort_values(by=['size', 'Task'], ascending=[True, False])


def group_results(df, group, add_const_num_cols=[]):
    string_cols = list(df.select_dtypes(include=['object']).columns)
    num_cols = df.select_dtypes(include=['number']).columns
    const_num_cols = ['size', 'shape-ratio'] + add_const_num_cols
    calc_cols = list(set(num_cols) - set(const_num_cols))
    base_grouping = ['dataset', 'Task', 'resources', 'framework', 'sparse_mode', 'sparse_amount']
    grouped = df.groupby(by=base_grouping + [group], as_index=False)[calc_cols].agg([np.median, np.mean])
    grouped.columns = list(map('_'.join, grouped.columns.values))
    grouped['count'] = df.groupby(by=base_grouping + [group]).size().values
    # Add constant cols and string cols with metadata
    const_cols = string_cols + const_num_cols
    grouped = pd.merge(grouped, df[const_cols].drop_duplicates(), on=base_grouping + [group], how='left')

    return grouped


def get_best_results(df_all, df_grouped, modi=['mode', 'mtl_mode'], target='relative_score_lit_median', optimum='min'):
    base_grouping = ['dataset', 'Task', 'resources', 'sparse_mode', 'sparse_amount']
    best_of_type_list = [f'best-of-{x}' for x in modi]
    for x in modi:
        df_grouped[f'best-of-{x}'] = 0
        if optimum == 'min':
            df_grouped.loc[df_grouped.groupby(base_grouping + [x])[target].idxmin(), f'best-of-{x}'] = 1
        elif optimum == 'max':
            df_grouped.loc[df_grouped.groupby(base_grouping + [x])[target].idxmax(), f'best-of-{x}'] = 1

    df_all = pd.merge(df_all, df_grouped[base_grouping + best_of_type_list + ['framework', 'mode']], on=base_grouping + ['framework', 'mode'], how='left')
    return df_all


def get_task_span(datasets=['guo-reduced-1000', 'hu', 'huang', 'xiong', 'yin'],
                  data_folder=os.path.join('..', 'data', 'storage')):
    df_span = pd.DataFrame(columns=['dataset', 'Task', 'span'])

    for dataset in datasets:
        path = os.path.join(data_folder, dataset)
        with open(os.path.join(path, 'y.txt'), 'r') as f:
            y = f.read().rstrip('\n').split(',')
        for task in y:
            data = pd.read_csv(os.path.join(path,'data.csv'), sep=';')
            y_df = data[task]
            span = float(y_df.max())- float(y_df.min())
            row = {'dataset':dataset.split('-')[0], 'Task':task, 'span':span}
            df_span.loc[len(df_span)] = row

    df_span['dataset-task-key'] = df_span['dataset'] + '_' + df_span['Task']
    df_span['dataset-task-key'] = df_span['dataset-task-key'].str.replace('guo-reduced-1000', 'guo')
    df_span.to_csv('span.csv')
    span_dict = pd.Series(df_span.span.values, index=df_span['dataset-task-key']).to_dict()
    return df_span, span_dict


def merge_with_correlations(results_df, correlation_df, aggregation_method='mean'):
    """
    Merges results_df with correlation_df and includes separate columns for each 
    correlation metric, containing either the max or mean of correlations to other tasks.
    
    Args:
    - results_df (pd.DataFrame): DataFrame with 'Dataset', 'Task', and 'r2' columns.
    - correlation_df (pd.DataFrame): DataFrame with 'Dataset', 'Correlation_Type', 'Column_Pair', and 'Value' columns.
    - aggregation_method (str): Method to aggregate correlations ('mean' or 'max').
    
    Returns:
    - merged_df (pd.DataFrame): Merged DataFrame with separate correlation columns added.
    """
    
    # Function to aggregate correlations based on the specified method
    def get_aggregated_correlation(correlation_values):
        if aggregation_method == 'mean':
            return abs(correlation_values).mean()
        elif aggregation_method == 'max':
            return correlation_values.max()
        else:
            raise ValueError("aggregation_method must be either 'mean' or 'max'")
    
    # Initialize a dictionary to hold aggregated correlation lists
    aggregated_correlations = {corr_type: [] for corr_type in correlation_df['correlation_type'].unique()}
    
    # Iterate over each unique dataset and task in the results_df
    for dataset, task in zip(results_df['dataset'], results_df['Task']):
        # Filter correlations for the current dataset and task
        dataset_correlations = correlation_df[correlation_df['dataset'] == dataset]
        
        # Loop through each unique correlation type
        for corr_type in correlation_df['correlation_type'].unique():
            # Filter for the current correlation type and task
            task_correlations = dataset_correlations[
                (dataset_correlations['correlation_type'] == corr_type) & 
                (dataset_correlations['column_pair'].apply(lambda pair: task in pair))
            ]
            
            # Extract the correlation values to the other tasks
            other_task_corrs = task_correlations['value']
            
            # Aggregate the correlation values
            aggregated_corr = get_aggregated_correlation(other_task_corrs)
            
            # Store the aggregated correlation
            aggregated_correlations[corr_type].append(aggregated_corr)
    
    # Add the aggregated correlations as columns to the results_df
    for corr_type in aggregated_correlations:
        results_df[f'{corr_type}_aggregated'] = aggregated_correlations[corr_type]
    
    return results_df
    

def plot_tasks(df, tasks=None, group='framework_type', target='relative_score', plot_dict=None, save_fig=False,
                x_range=None):
    layout = go.Layout(
        xaxis=dict(title=target, title_font={'size': 24}, tickfont={'size': 20},
                   zeroline=False, linecolor='black', gridcolor='#cccccc'),
        yaxis=dict(linecolor='black', title_font={'size': 24}, tickfont={'size': 20}),
        boxmode='group',
        plot_bgcolor='white',
        # xaxis_title=target.replace('_',' '),
        legend=dict(traceorder='reversed', font_size=22, orientation="v",
                    y=0.55, xanchor='center', x=1.1)
    )

    fig = go.Figure(layout=layout)
    if tasks:
        df = df[df['dataset-task-key'].isin(tasks)]

    for framework_name, framework_dict in plot_dict.items():
        fig.add_trace(go.Box(
            x=df[target].loc[df[group] == framework_name],
            y=df['dataset-task-key'].loc[df[group] == framework_name],
            name=framework_name,
            boxmean=False,
            #fillcolor=framework_dict['fill-colour'],
            # line_color='black',
            line_width=2,
            #boxpoints='all',
            marker_color=framework_dict['colour']
        ))

    if 'relative' in target:
        fig.add_vline(x=1, line_color="black")
    else:
        for Task in df['dataset-task-key'].unique():
            max_value = df_lit['r2'].loc[df_lit['Task'] == Task].max()
            fig.add_trace(go.Scatter(mode='markers', x=[max_value], y=[Task], marker_symbol='line-ns',
                                     marker_line_color="midnightblue", marker_color="lightskyblue",
                                     marker_line_width=3, marker_size=20, showlegend=False))

    # literature range
    lit_colour = 'black'
    for counter, lit in enumerate((LIT_DICT.keys() & df['Task'].unique())):
        if counter == 0:
            legend_indicator = True
        else:
            legend_indicator = False
        range_ = LIT_DICT[lit]['range']
        fig.add_trace(go.Scatter(x=range_, y=[lit, lit], mode='lines', line_width=5, line_color=lit_colour,
                                 name='literature', showlegend=legend_indicator))
        fig.add_trace(
            go.Scatter(mode='markers', x=[range_[0]], y=[lit], marker_symbol='line-ns', marker_line_color=lit_colour,
                       marker_line_width=5, marker_size=10, marker_color=lit_colour,
                       showlegend=False))
        fig.add_trace(
            go.Scatter(mode='markers', x=[range_[1]], y=[lit], marker_symbol='line-ns', marker_line_color=lit_colour,
                       marker_line_width=5, marker_size=10, marker_color=lit_colour,
                       showlegend=False))

    # fig.update_traces(width=1, selector=dict(type='box'))
    fig.update_traces(orientation='h')  # horizontal box plots
    fig.update_traces(whiskerwidth=1, selector=dict(type='box'))
    if x_range:
        fig.update_layout(xaxis_range=x_range)
    # fig.update_layout(width=1100, height=350)
    # fig.update_layout(xaxis=dict(tick0=0, dtick=0.25))
    # fig.update_layout(boxgap=0.15, boxgroupgap=0.4)
    if save_fig:
        fig.write_image(save_fig, scale=1)
    #fig.show()
    return fig

def plot_summarize(df, tasks=None, group='mode',
                   target='relative_score_stl_median', plot_dict=None, save_fig=False, y_range=None):
    df['sparse_key'] = df['sparse_mode'] + df['sparse_amount'].astype(str)
    layout = go.Layout(yaxis=dict(title=target, title_font={'size': 20}, tickfont={'size': 16},
                                  zeroline=False, linecolor='black', gridcolor='#cccccc'),
                       xaxis=dict(title=None, linecolor='black', title_font={'size': 20}, tickfont={'size': 16}),
                       boxmode='group',
                       plot_bgcolor='white',
                       legend=dict(font_size=18, orientation="h", yanchor="top", y=1.1, xanchor='center', x=0.5)
    )

    fig = go.Figure(layout=layout)

    for group_, group_specs in plot_dict.items():
        fig.add_trace(go.Box(
            y=df[target].loc[df[group] == group_],
            x=df['sparse_key'].loc[df[group] == group_],
            name=group_,
            marker_color=group_specs['colour'],
            # fillcolor=framework_dict['colour'],
            # line_color='black',
            line_width=2,
            #boxpoints='all'
        ))

    fig.add_hline(y=1, line_color="black")
    fig.update_traces(whiskerwidth=1, selector=dict(type='box'))
    fig.update_traces(quartilemethod="exclusive")  # or "inclusive", or "linear" by default
    if y_range:
        fig.update_layout(yaxis_range=y_range)
    if save_fig:
        fig.write_image(save_fig)
    #fig.show()
    return fig

def plot_box(df_plot, y_axis, x_axis, group, target='relative_score_stl_median', plot_dict=None, save_fig=False, y_range=None):
    layout = go.Layout(yaxis=dict(title=target, title_font={'size': 20}, tickfont={'size': 16},
                                  zeroline=False, linecolor='black'),
                       xaxis=dict(title=None, linecolor='black', title_font={'size': 20}, tickfont={'size': 16}, gridcolor='#cccccc'),
                       boxmode='group',
                       plot_bgcolor='white',
                       legend=dict(font_size=18, orientation="h", yanchor="top", y=1.1, xanchor='center', x=0.5, traceorder='reversed')
    )

    fig = go.Figure(layout=layout)

    for mode, mode_specs in eval(f'dict_{group}').items():
        fig.add_trace(go.Box(
            y=df_plot[y_axis].loc[df_plot[group] == mode],
            x=df_plot[x_axis].loc[df_plot[group] == mode],
            name=mode,
            marker_color=mode_specs['colour'],
            # fillcolor=framework_dict['colour'],
            # line_color='black',
            line_width=1.5,
            # boxpoints='all'
        ))

    #fig.add_hline(y=1, line_color="black")
    fig.update_traces(whiskerwidth=1, selector=dict(type='box'))
    fig.update_traces(quartilemethod="exclusive")  # or "inclusive", or "linear" by default
    if y_range:
        fig.update_layout(yaxis_range=y_range)
    if save_fig:
        fig.write_image(save_fig)
    #fig.show()
    return fig


def plot_box_binned(df_plot, y_axis, x_axis, group, bins=5, custom_bins=None, color_dict=None):
    # Bin the x-axis values
    if color_dict is None:
        color_dict = eval(f'dict_{group}')
    if custom_bins is not None:
        bins = np.array(custom_bins)  # Custom bins
    df_plot['x_binned'] = pd.cut(df_plot[x_axis], bins=bins)
    
    layout = go.Layout(
        yaxis=dict(
            title={'text': y_axis, 'font': {'size': 22}}, tickfont={'size': 18}, zeroline=False, linecolor='black', gridcolor='#cccccc'),
        xaxis=dict(
            title={'text': f'{x_axis} (binned)', 'font': {'size': 22}}, tickfont={'size': 18}, linecolor='black', gridcolor='#cccccc'),
        plot_bgcolor='white',
        boxmode='group', 
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        legend=dict(font_size=22)
    )

    fig = go.Figure(layout=layout)

    # Add traces for each group
    for mode, mode_specs in color_dict.items():
        df_filtered = df_plot[df_plot[group] == mode]

        fig.add_trace(go.Box(
            x=df_filtered['x_binned'].astype(str),
            y=df_filtered[y_axis],
            name=mode,
            marker=dict(color=mode_specs['colour']),
        ))

    return fig

def plot_scatter(df_plot, y_axis, x_axis, group, z_color=None):
    # Check if z_color is a Pandas Series or a DataFrame with one column
    if isinstance(z_color, pd.DataFrame) and z_color.shape[1] == 1:
        z_color = z_color.iloc[:, 0]  # Convert to a Series by selecting the single column
    elif not isinstance(z_color, (pd.Series, type(None))):
        raise ValueError("z_color must be a Pandas Series, a single-column DataFrame, or None.")

    
    layout = go.Layout(yaxis=dict(title={'text':y_axis, 'font':{'size':22}}, tickfont={'size':18},                         
                                  zeroline=False, linecolor='black', gridcolor='#cccccc'),
                       xaxis=dict(title={'text':x_axis, 'font':{'size':22}}, linecolor='black', tickfont={'size':18}, gridcolor='#cccccc'),
                       plot_bgcolor='white',
                       boxmode='group',
                       colorway=['#00305e'],
                       margin ={"l":10,"r":10,"t":10,"b":10},
                       legend=dict(font_size=22)# ,orientation="h", yanchor="top",y=1.1,xanchor='center',x=0.5)
                      )
   
    fig = go.Figure(layout=layout)
 
    # Add traces for each group
    for mode, mode_specs in eval(f'dict_{group}').items():
        df_filtered = df_plot[df_plot[group] == mode]
        z_color_filtered = z_color.loc[df_filtered.index].tolist() if z_color is not None else None

        marker_dict = dict(size=8, color=z_color_filtered or mode_specs['colour'])
        # Add additional marker parameters if z_color is provided
        if z_color is not None:
            marker_dict.update({
                'colorscale': 'Plasma',
                'showscale': True
            })
            
        fig.add_trace(go.Scatter(
            y=df_filtered[y_axis],
            x=df_filtered[x_axis],
            mode='markers',
            name=mode,
            marker=marker_dict
        ))
    
    return fig

def power_law(x, m, p):
    return m * np.power(x,p)

def fit_power_law(df, target, GROUP, p0=(0.67, -0.324), maxfev=40000000):
    df = df[~df[target].isna()]
    # Dictionary to store results
    fit_results = {}
    # Define x_set for plotting
    x_set = np.logspace(0.4, 2, num=2000)
    for group in df[GROUP].unique():
        df_fit = df[df[GROUP] == group]
        #Shape
        params_shape, cv_shape = curve_fit(power_law, df_fit['shape-ratio'], df_fit[target], p0=p0, maxfev=maxfev)
        y_set_shape = power_law(x_set, *params_shape)
        r2_shape = r2_score(df_fit[target], power_law(df_fit['shape-ratio'], *params_shape))
        #Size
        params_size, cv_size = curve_fit(power_law, df_fit['size'], df_fit[target], p0=p0, maxfev=maxfev)
        y_set_size = power_law(x_set, *params_size)
        r2_size = r2_score(df_fit[target], power_law(df_fit['shape-ratio'], *params_size))
        # Save to dict
        fit_results[group] = {'params_shape': params_shape, 'cv_shape': cv_shape, 'x_set_shape': x_set, 'y_set_shape': y_set_shape, 'r2_shape': r2_shape,
                                 'params_size': params_size, 'cv_size': cv_size, 'x_set_size': x_set, 'y_set_size': y_set_size, 'r2_size': r2_size
                                }
        # Print results
        print(f'Params Fit over dataset shape for mtl_mode={group}:', params_shape)
        print('R2 Shape-Ratio: ',r2_shape)
        print(f'Params Fit over dataset size for mtl_mode={group}:', params_size)
        print('R2 Size: ',r2_size)
    return fit_results
