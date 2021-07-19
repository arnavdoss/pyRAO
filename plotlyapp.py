# -*- coding: utf-8 -*-
import dash  # pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import capytaine as cpt  # conda install -c conda-forge capytaine
import numpy as np
from Solver.EOM import EOM
from Solver.meshmaker import meshmaker
from Solver.JONSWAP import response
import dash_bootstrap_components as dbc  # conda install -c conda-forge dash-bootstrap-components
from dash.exceptions import PreventUpdate
from Solver.hydrostatics_wrapper import meshK

# from jupyter_dash import JupyterDash

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
# app = JupyterDash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

Aqua = '#00ADEF'
Navy = '#00306B'
Gray = '#EBE9E9'
Signal = 'DD1C1A'

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #EBE9E9',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #EBE9E9',
    'borderBottom': '1px solid #EBE9E9',
    'backgroundColor': Aqua,
    'color': 'white',
    'padding': '6px',
    'fontWeight': 'bold'
}

button_style = {
    'backgroundColor': 'white',
    'border': '0px solid #EBE9E9',
}

input_style = {
    'width': '100px',
    'border': '1px solid #EBE9E9',
    'textAlign': 'center'
}

app.layout = dbc.Container(
    [
        dbc.Row([
            dcc.Interval(id='interval', n_intervals=0, interval=1000),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Img(
                                    src=app.get_asset_url('python-logo.svg'),
                                    alt="Python Logo", width="70dp"),
                            ], width=2, align="start"),
                            dbc.Col([
                                html.H3('pyRAO', style={'color': Aqua}),
                                html.H6('Open-source diffraction app with Capytaine, Meshmagick & Dash',
                                        style={'color': Aqua})
                            ], width=10, align="start"),
                        ])
                    ])
                ]),
                dbc.Row([
                    html.H1(" ")
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupAddon(
                                html.Button(id='run_button', style=button_style, children=[
                                    html.Img(height='50px ', title='Begin diffraction analysis',
                                             src=app.get_asset_url('start_icon.svg'))]), addon_type="prepend"
                            ), dbc.Progress(id='progress_bar',
                                            style={"height": "40px", "width": "90%", 'position': 'absolute',
                                                   'top': '10%', 'right': '0%'}),
                        ], style={'width': '100%'})
                    ], width=12)
                ]),
                dbc.Row([
                    html.H1(" ")
                ]),
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            dbc.Row(form=True, children=[
                                dbc.Col([
                                    dbc.FormGroup([
                                        html.H6('Barge global dimensions')
                                    ])
                                ], width=3),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Input(id='v_l', type='number', value=122, persistence=False, bs_size='sm',
                                                  persistence_type='local', inputMode='numeric', style=input_style),
                                        dbc.Label('Length [m]', html_for='v_l', size='sm')
                                    ]),

                                ], width=2),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Input(id='v_b', type='number', value=32, persistence=False, bs_size='sm',
                                                  persistence_type='local', inputMode='numeric', style=input_style),
                                        dbc.Label('Beam [m]', html_for='v_b', size='sm')
                                    ]),
                                ], width=2),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Input(id='v_h', type='number', value=8, persistence=False, bs_size='sm',
                                                  persistence_type='local', inputMode='numeric', style=input_style),
                                        dbc.Label('Height [m]', html_for='v_h', size='sm')
                                    ]),
                                ], width=2),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Input(id='v_t', type='number', value=4, persistence=False, bs_size='sm',
                                                  persistence_type='local', inputMode='numeric', style=input_style),
                                        dbc.Label('Draft [m]', html_for='v_t', size='sm')
                                    ]),
                                ], width=2)
                            ]),
                            dbc.Row(form=True, children=[
                                dbc.Col([
                                    dbc.FormGroup([
                                        html.H6('Vessel COG')
                                    ])
                                ], width=3),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Input(id='cogx', type='number', value=0, persistence=False, bs_size='sm',
                                                  persistence_type='local', inputMode='numeric', disabled=True,
                                                  style=input_style),
                                        dbc.Label('LCG [m]  Midship 0', html_for='cogx', size='sm')
                                    ]),
                                ], width=2),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Input(id='cogy', type='number', value=0, persistence=False, bs_size='sm',
                                                  persistence_type='local', inputMode='numeric', disabled=True,
                                                  style=input_style),
                                        dbc.Label('TCG [m]  Centerline 0', html_for='cogy', size='sm')
                                    ]),
                                ], width=2),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Input(id='cogz', type='number', value=10, persistence=False, bs_size='sm',
                                                  persistence_type='local', inputMode='numeric', style=input_style),
                                        dbc.Label('VCG [m]   Baseline 0', html_for='cogz', size='sm')
                                    ]),
                                ], width=2),
                            ]),
                            dbc.Row(form=True, children=[
                                dbc.Col([
                                    dbc.FormGroup([
                                        html.H6('Mesh panel dimensions')
                                    ])
                                ], width=3),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Input(id='p_l', type='number', value=4, persistence=False, bs_size='sm',
                                                  persistence_type='local', inputMode='numeric', style=input_style),
                                        dbc.Label('Length [m]', html_for='p_l', size='sm')
                                    ]),
                                ], width=2),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Input(id='p_w', type='number', value=4, persistence=False, bs_size='sm',
                                                  persistence_type='local', inputMode='numeric', style=input_style),
                                        dbc.Label('Width [m]', html_for='p_w', size='sm')
                                    ]),
                                ], width=2),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Input(id='p_h', type='number', value=2, persistence=False, bs_size='sm',
                                                  persistence_type='local', inputMode='numeric', style=input_style),
                                        dbc.Label('Height [m]', html_for='p_h', size='sm')
                                    ]),
                                ], width=2),
                            ]),
                            dbc.Row(form=True, children=[
                                dbc.Col([
                                    dbc.FormGroup([
                                        html.H6('Waves')
                                    ])
                                ], width=3),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Input(id='d_min', type='number', value=0, persistence=False, bs_size='sm',
                                                  persistence_type='local', inputMode='numeric', min=0, max=360,
                                                  step=15, style=input_style),
                                        dbc.Label('Direction [deg]', html_for='d_min', size='sm')
                                    ]),
                                ], width=2),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Input(id='t_min', type='number', value=5, persistence=False, bs_size='sm',
                                                  persistence_type='local', inputMode='numeric', min=1,
                                                  style=input_style),
                                        dbc.Label('Minimum period [s]', html_for='t_min', size='sm')
                                    ]),
                                ], width=2),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Input(id='t_max', type='number', value=20, persistence=False, bs_size='sm',
                                                  persistence_type='local', inputMode='numeric', min=1,
                                                  style=input_style),
                                        dbc.Label('Maximum period [s]', html_for='t_max', size='sm')
                                    ]),
                                ], width=2),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Input(id='n_t', type='number', value=20, persistence=False, bs_size='sm',
                                                  persistence_type='local', inputMode='numeric', min=1,
                                                  style=input_style),
                                        dbc.Label('No. of periods', html_for='n_t', size='sm')
                                    ]),
                                ], width=2),
                            ]),
                            dbc.Row(form=True, children=[
                                dbc.Col([
                                    dbc.FormGroup([
                                        html.H6('Water properties')
                                    ])
                                ], width=3),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Input(id='water_depth', type='number', value=500, persistence=False,
                                                  bs_size='sm',
                                                  persistence_type='local', inputMode='numeric', style=input_style),
                                        dbc.Label('Depth [m]', html_for='water_depth', size='sm')
                                    ]),

                                ], width=2),
                                dbc.Col([
                                    dbc.FormGroup([
                                        dbc.Input(id='rho_water', type='number', value=1025, persistence=False,
                                                  bs_size='sm',
                                                  persistence_type='local', inputMode='numeric', disabled=True,
                                                  style=input_style),
                                        dbc.Label('Density [kg/m^3]', html_for='rho_water', size='sm')
                                    ]),
                                ], width=3),
                            ]),
                        ])
                    ])
                ]),
            ], width=5),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Col([
                            html.Div([
                                dcc.Tabs(id="tabs-styled-with-inline", value='tab-1', children=[
                                    dcc.Tab(label='RAO Plot', value='tab-1', style=tab_style,
                                            selected_style=tab_selected_style, children=[
                                            html.Div(id='wrapper_div', children=[
                                                dcc.Store(id='RAO_data'),
                                                dcc.Store(id='Value_data'),
                                                dcc.Graph(id='graph', style={'height': '70vh'}),
                                                # dcc.Graph(id='graph_FRAO', style={'height': '70vh'}),
                                            ]),
                                        ]),
                                    dcc.Tab(label='Hydrostatics Report', value='tab-2', style=tab_style,
                                            selected_style=tab_selected_style, children=[
                                            dash_table.DataTable(
                                                id='HS_report', style_cell={'textAlign': 'left'},
                                                style_as_list_view=True),
                                        ]),
                                    dcc.Tab(label='Response Plot', value='tab-3', style=tab_style,
                                            selected_style=tab_selected_style, children=[
                                            dbc.Row([
                                                html.H1(' ')
                                            ]),
                                            dbc.Row(form=True, children=[
                                                dbc.Col([
                                                    dbc.FormGroup([
                                                        html.H6('Wave properties')
                                                    ])
                                                ], width=3),
                                                dbc.Col([
                                                    dbc.FormGroup([
                                                        dbc.Input(id='Hs', type='number', value=2, persistence=False,
                                                                  bs_size='sm', persistence_type='local', min=0,
                                                                  inputMode='numeric', style=input_style),
                                                        dbc.Label('Wave Height [m]', html_for='Hs', size='sm')
                                                    ]),

                                                ], width=2),
                                                dbc.Col([
                                                    dbc.FormGroup([
                                                        dbc.Input(id='Tp', type='number', value=5, persistence=False,
                                                                  bs_size='sm', persistence_type='local',
                                                                  inputMode='numeric', min=0, disabled=True,
                                                                  style=input_style),
                                                        dbc.Label('Peak period [s]', html_for='Tp', size='sm')
                                                    ]),
                                                ], width=2),
                                                dbc.Col([
                                                    dbc.FormGroup([
                                                        dbc.Input(id='gamma', type='number', value=3.3,
                                                                  persistence=False,
                                                                  bs_size='sm', persistence_type='local',
                                                                  inputMode='numeric', min=0, disabled=True,
                                                                  style=input_style),
                                                        dbc.Label('Gamma [s]', html_for='gamma', size='sm')
                                                    ]),
                                                ], width=2),
                                            ]),
                                            dbc.Row([
                                                dcc.Graph(
                                                    id='response_graph', style={'height': '70vh'}
                                                )
                                            ])
                                        ]),
                                ], style=tabs_styles),
                                html.Div(id='tabs-content-inline')
                            ])
                        ], width=12)
                    ])
                ])
            ], width=7)
        ]),
    ], fluid=True, style={'padding-top': '15px'})


@app.callback([Output('Value_data', 'data'), Output('RAO_data', 'data')], [
    Input('run_button', 'n_clicks'),
    Input('v_l', 'value'), Input('v_b', 'value'), Input('v_h', 'value'),
    Input('v_t', 'value'), Input('cogx', 'value'), Input('cogy', 'value'), Input('cogz', 'value'),
    Input('p_l', 'value'), Input('p_w', 'value'), Input('p_h', 'value'), Input('t_min', 'value'),
    Input('t_max', 'value'), Input('n_t', 'value'), Input('d_min', 'value'), Input('water_depth', 'value'),
    Input('rho_water', 'value')
])
def initialize_value(n_clicks, v_l, v_b, v_h, v_t, cogx, cogy, cogz, p_l, p_w, p_h, t_min, t_max, n_t, d_min,
                     water_depth, rho_water):
    ctx = dash.callback_context
    if ctx.triggered:
        trigger_name = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_name == 'run_button':
        Values = makeValues(v_l, v_b, v_h, v_t, cogx, cogy, cogz, p_l, p_w, p_h, t_min, t_max, n_t, d_min, water_depth,
                            rho_water)
        RAOs = {
            'Period': 0,
            'Surge': 0,
            'Sway': 0,
            'Heave': 0,
            'Roll': 0,
            'Pitch': 0,
            'Yaw': 0,
        }
        Valuespd = pd.DataFrame.from_records([Values])
        Values_json = Valuespd.to_json()
        RAOpd = pd.DataFrame.from_records([RAOs])
        RAOs_json = RAOpd.to_json()
        global body, Mk, Ck
        body, Mk, Ck, HS_report = makemesh(Valuespd)
        return [Values_json, RAOs_json]


# #         RAOpd, FRAO = initialize_calc(Values)
# #         columns = [{"name": i, "id": i} for i in RAOpd.columns]
# #         data = RAOpd.to_dict('records')
# #         style_data_conditional = [
# #             {
# #                 'if': {'row_index': 'odd'},
# #                 'backgroundColor': Gray}
# #         ]
# #         graph = create_figure(RAOpd)
# #         RAO_json = RAOpd.to_json()
# #         graph_FRAO = create_figure(FRAO)
# #         graph_FRAO.update_layout(title_text='Barge Force RAO', yaxis=dict(showexponent='all', exponentformat='e'))
# #         graph_FRAO.update_xaxes(title_text='Period [s]')
# #         graph_FRAO.update_yaxes(title_text='Translational Force RAOs [N/m]', secondary_y=False)
# #         graph_FRAO.update_yaxes(title_text='Rotational Force RAOs [N.rad/m]', secondary_y=True)
# #         return [graph, columns, data, style_data_conditional, RAO_json, graph_FRAO]


@app.callback([Output('wrapper_div', 'children'), Output('progress_bar', 'value'), Output('progress_bar', 'children')],
              [Input('Value_data', 'data'), Input('RAO_data', 'data'), Input('interval', 'n_intervals')])
def run_diff(Values_json, RAOpd_json, n):
    Valuespd_in = pd.read_json(Values_json)
    RAOpd_in = pd.read_json(RAOpd_json)
    omega = np.linspace(float(Valuespd_in["t_min"]), float(Valuespd_in["t_max"]), int(Valuespd_in["n_t"]))
    inputs = Valuespd_in.copy()
    inputs['n_t'] = 1
    count = Valuespd_in['counter']
    inputs['t_min'] = omega[count].tolist()[0]
    if float(count) < float(Valuespd_in['n_t']) + 1:
        RAO_val, FRAO_val = EOM(body, Mk, Ck, inputs, show=False).solve()
        RAO_val.insert(0, omega[count].tolist()[0])
        RAOpd_in.loc[len(RAOpd_in)] = RAO_val
        Valuespd_in['counter'] = float(Valuespd_in['counter']) + 1
        figure_RAO = create_figure(RAOpd_in)
        RAOpd_out = RAOpd_in.to_json()
        Values_out = Valuespd_in.to_json()
        progress = np.rint(((float(count) + 1) / float(Valuespd_in['n_t'])) * 100)
        return [[
            dcc.Store(id='RAO_data', data=RAOpd_out),
            dcc.Store(id='Value_data', data=Values_out),
            dcc.Graph(id='graph', figure=figure_RAO, style={'height': '70vh'})],
            progress, f"{progress} %" if progress >= 5 else ""
        ]
    else:
        raise PreventUpdate


def create_figure(RAOpd):
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(go.Scatter(name='Surge', x=RAOpd["Period"].tolist(), y=RAOpd["Surge"].tolist()),
                     secondary_y=False, )
    figure.add_trace(go.Scatter(name='Sway', x=RAOpd["Period"].tolist(), y=RAOpd["Sway"].tolist()),
                     secondary_y=False, )
    figure.add_trace(go.Scatter(name='Heave', x=RAOpd["Period"].tolist(), y=RAOpd["Heave"].tolist()),
                     secondary_y=False, )
    figure.add_trace(go.Scatter(name='Roll', x=RAOpd["Period"].tolist(), y=np.rad2deg(RAOpd["Roll"].tolist())),
                     secondary_y=True, )
    figure.add_trace(go.Scatter(name='Pitch', x=RAOpd["Period"].tolist(), y=np.rad2deg(RAOpd["Pitch"].tolist())),
                     secondary_y=True, )
    figure.add_trace(go.Scatter(name='Yaw', x=RAOpd["Period"].tolist(), y=np.rad2deg(RAOpd["Yaw"].tolist())),
                     secondary_y=True, )
    figure.update_layout(title_text='Barge RAO', yaxis=dict(showexponent='all', exponentformat='e'))
    figure.update_xaxes(title_text='Period [s]',
                        range=[sorted(RAOpd['Period'].tolist())[1], max(RAOpd['Period'].tolist())])
    figure.update_yaxes(title_text='Translational RAOs [m/m]', secondary_y=False)
    figure.update_yaxes(title_text='Rotational RAOs [deg/m]', secondary_y=True)
    figure.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return figure


def makemesh(a):
    mesh = meshmaker(a["v_l"], a["v_b"], 0, a["v_t"], a["p_l"], a["p_w"], a["p_h"])
    faces, vertices = mesh.barge()
    mesh = cpt.Mesh(vertices=vertices, faces=faces)
    body = cpt.FloatingBody(mesh=mesh, name="barge")
    mesh2 = meshmaker(a["v_l"], a["v_b"], a["v_h"], a["v_t"], a["p_l"], a["p_w"], a["p_h"])
    faces2, vertices2 = mesh2.barge()
    Mk, Ck, HS_report = meshK(faces2, vertices2, float(a['cogx']), float(a['cogy']), float(a['cogz']),
                              float(a['rho_water']), 9.81)
    return body, Mk, Ck, HS_report


def makeValues(v_l, v_b, v_h, v_t, cogx, cogy, cogz, p_l, p_w, p_h, t_min, t_max, n_t, d_min, water_depth, rho_water):
    Values = {
        'v_l': v_l,
        'v_b': v_b,
        'v_h': v_h,
        'v_t': v_t,
        'cogx': cogx,
        'cogy': cogy,
        'cogz': cogz,
        'p_l': p_l,
        'p_w': p_w,
        'p_h': p_h,
        't_min': t_min,
        't_max': t_max,
        'n_t': n_t,
        'd_min': d_min,
        'd_max': 0,
        'n_d': 1,
        'water_depth': water_depth,
        'rho_water': rho_water,
        'counter': 0
    }
    return Values


@app.callback([Output('HS_report', 'data'), Output('HS_report', 'columns')], [
    Input('v_l', 'value'), Input('v_b', 'value'), Input('v_h', 'value'),
    Input('v_t', 'value'), Input('cogx', 'value'), Input('cogy', 'value'), Input('cogz', 'value'),
    Input('p_l', 'value'), Input('p_w', 'value'), Input('p_h', 'value'), Input('t_min', 'value'),
    Input('t_max', 'value'), Input('n_t', 'value'), Input('d_min', 'value'), Input('water_depth', 'value'),
    Input('rho_water', 'value')
])
def HSReport(v_l, v_b, v_h, v_t, cogx, cogy, cogz, p_l, p_w, p_h, t_min, t_max, n_t, d_min, water_depth, rho_water):
    Values = makeValues(v_l, v_b, v_h, v_t, cogx, cogy, cogz, p_l, p_w, p_h, t_min, t_max, n_t, d_min, water_depth,
                        rho_water)
    body, Mk, Ck, HS_report = makemesh(Values)
    HS_report = HS_report.splitlines()
    output = [html.P('\n \n \n')]
    for a in range(len(HS_report)):
        curline = HS_report[a].replace('-', '').replace('\t', '').replace('**', '^').replace('>', '          ').split(
            '          ')
        if len(curline) == 1:
            curline.append(' ')
        output.append(curline)
    data = pd.DataFrame(output, columns=['Parameter', 'Value'])
    columns = [{'id': c, 'name': c} for c in data.columns]
    data = data.to_dict('records')
    return [data, columns]


if __name__ == '__main__':
    app.run_server(debug=True)
