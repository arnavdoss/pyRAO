# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import capytaine as cpt
import numpy as np
from EOM import EOM
from meshmaker import meshmaker
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
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
    'border': '1px solid #EBE9E9'
}

input_style = {
    'width': '100px',
    'border': '1px solid #EBE9E9',
    'textAlign': 'center'
}

app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Img(
                                    src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg",
                                    alt="Python Logo", width="75%"),
                            ], width=1, align="start"),
                            dbc.Col([
                                html.H1('pyRAO', style={'color': Aqua}),
                                html.H5('Open-source diffraction app with Capytaine & Dash', style={'color': Aqua})
                            ], width=4, align="start"),
                        ], style={'height': '10vh'})
                    ])
                ]),
            ], width=12),
        ], align='end'),
        dbc.Row([
            html.H1(' ')
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H4("Run button", style={'color': Navy}),
                            html.Button('Run Diffraction', id='run_button', style=button_style)
                        ]),
                    ], style={'height': '10vh'})
                ]),
                dbc.Row([
                    html.H1(" ")
                ]),
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H4("Inputs", style={'color': Navy}),
                            dbc.Row([
                                dbc.Col([
                                    html.H6('Vessel dimensions')
                                ], width=3),
                                dbc.Col([
                                    dcc.Input(id='v_l', type='number', value=122, persistence=True,
                                              persistence_type='local',
                                              inputMode='numeric', style=input_style)
                                ], width=2),
                                dbc.Col([
                                    dcc.Input(id='v_b', type='number', value=32, persistence=True,
                                              persistence_type='local',
                                              inputMode='numeric', style=input_style)
                                ], width=2),
                                dbc.Col([
                                    dcc.Input(id='v_h', type='number', value=8, persistence=True,
                                              persistence_type='local',
                                              inputMode='numeric', style=input_style)
                                ], width=2),
                                dbc.Col([
                                    dcc.Input(id='v_t', type='number', value=4.9, persistence=True,
                                              persistence_type='local',
                                              inputMode='numeric', style=input_style)
                                ], width=2)
                            ]),
                            dbc.Row([
                                dbc.Col([html.P(' ')], width=3),
                                dbc.Col([html.P('Length [m]')], width=2),
                                dbc.Col([html.P('Breadth [m]')], width=2),
                                dbc.Col([html.P('Height [m]')], width=2),
                                dbc.Col([html.P('Draft [m]')], width=2),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H6('Vessel COG')
                                ], width=3),
                                dbc.Col([
                                    dcc.Input(id='cogx', type='number', value=0, persistence=True,
                                              persistence_type='local',
                                              inputMode='numeric', style=input_style)
                                ], width=2),
                                dbc.Col([
                                    dcc.Input(id='cogy', type='number', value=0, persistence=True,
                                              persistence_type='local',
                                              inputMode='numeric', style=input_style)
                                ], width=2),
                                dbc.Col([
                                    dcc.Input(id='cogz', type='number', value=15, persistence=True,
                                              persistence_type='local',
                                              inputMode='numeric', style=input_style)
                                ], width=2)
                            ]),
                            dbc.Row([
                                dbc.Col([html.P(' ')], width=3),
                                dbc.Col([html.P('LCG [m]')], width=2),
                                dbc.Col([html.P('VCG [m]')], width=2),
                                dbc.Col([html.P('TCG [m]')], width=2)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H6('Panel dimensions')
                                ], width=3),
                                dbc.Col([
                                    dcc.Input(id='p_l', type='number', value=4, persistence=True,
                                              persistence_type='local',
                                              inputMode='numeric', style=input_style)
                                ], width=2),
                                dbc.Col([
                                    dcc.Input(id='p_w', type='number', value=4, persistence=True,
                                              persistence_type='local',
                                              inputMode='numeric', style=input_style)
                                ], width=2),
                                dbc.Col([
                                    dcc.Input(id='p_h', type='number', value=4, persistence=True,
                                              persistence_type='local',
                                              inputMode='numeric', style=input_style)
                                ], width=2)
                            ]),
                            dbc.Row([
                                dbc.Col([html.P(' ')], width=3),
                                dbc.Col([html.P('Length [m]')], width=2),
                                dbc.Col([html.P('Width [m]')], width=2),
                                dbc.Col([html.P('Height [m]')], width=2)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H6('Wave Periods')
                                ], width=3),
                                dbc.Col([
                                    dcc.Input(id='t_min', type='number', value=1, persistence=True,
                                              persistence_type='local',
                                              inputMode='numeric', style=input_style)
                                ], width=2),
                                dbc.Col([
                                    dcc.Input(id='t_max', type='number', value=20, persistence=True,
                                              persistence_type='local',
                                              inputMode='numeric', style=input_style)
                                ], width=2),
                                dbc.Col([
                                    dcc.Input(id='n_t', type='number', value=20, persistence=True,
                                              persistence_type='local',
                                              inputMode='numeric', step=1, style=input_style)
                                ], width=2)
                            ]),
                            dbc.Row([
                                dbc.Col([html.P(' ')], width=3),
                                dbc.Col([html.P('Min period [s]')], width=2),
                                dbc.Col([html.P('Max period [s]')], width=2),
                                dbc.Col([html.P('No. of periods')], width=2)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H6('Wave directions')
                                ], width=3),
                                dbc.Col([
                                    dcc.Input(id='d_min', type='number', value=0, persistence=True,
                                              persistence_type='local', style=input_style,
                                              inputMode='numeric', min=0, max=360, step=90)
                                ], width=2),
                                dbc.Col([
                                    dcc.Input(id='d_max', type='number', value=0, persistence=True,
                                              persistence_type='local', style=input_style,
                                              disabled=True, inputMode='numeric')
                                ], width=2),
                                dbc.Col([
                                    dcc.Input(id='n_d', type='number', value=1, persistence=True,
                                              persistence_type='local', style=input_style,
                                              disabled=True, inputMode='numeric', step=1)
                                ], width=2)
                            ]),
                            dbc.Row([
                                dbc.Col([html.P(' ')], width=3),
                                dbc.Col([html.P('Min dir [deg]')], width=2),
                                dbc.Col([html.P('Max dir [deg]')], width=2),
                                dbc.Col([html.P('No. of directions')], width=2)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H6('Water properties')
                                ], width=3),
                                dbc.Col([
                                    dcc.Input(id='water_depth', type='number', value=347.8, persistence=True,
                                              persistence_type='local', inputMode='numeric', style=input_style)
                                ], width=2),
                                dbc.Col([
                                    dcc.Input(id='rho_water', type='number', value=1025, persistence=True,
                                              persistence_type='local', disabled=True, inputMode='numeric',
                                              style=input_style)
                                ], width=2),
                            ]),
                            dbc.Row([
                                dbc.Col([html.P(' ')], width=3),
                                dbc.Col([html.P('Depth [m]')], width=2),
                                dbc.Col([html.P('Density [kg/m^3]')], width=2)
                            ]),
                        ])
                    ], style={'height': '60vh'})
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.P([
                                    html.A("Source code, ", href="https://github.com/arnavdoss/pyRAO",
                                           style={'color': Navy}),
                                    html.A("Plot.ly & ", href="https://plotly.com/", style={'color': Navy}),
                                    html.A("Dash", href="https://plotly.com/dash/", style={'color': Navy})
                                ]),
                            ], style={'height': '8vh'})
                        ])
                    ], style={'width': 12, 'padding': 15})
                ])
            ], width=5),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Col([
                            html.Div([
                                dcc.Tabs(id="tabs-styled-with-inline", value='tab-1', children=[
                                    dcc.Tab(label='RAO Plot', value='tab-1', style=tab_style,
                                            selected_style=tab_selected_style, children=[
                                                dcc.Graph(
                                                    id='graph', style={'height': '70vh'}
                                                )
                                            ]),
                                    dcc.Tab(label='Output Table', value='tab-2', style=tab_style,
                                            selected_style=tab_selected_style, children=[
                                                dash_table.DataTable(
                                                    id='table',
                                                )
                                            ]),
                                ], style=tabs_styles),
                                html.Div(id='tabs-content-inline')
                            ])
                        ], width=12)
                    ], style={'height': '81vh', 'overflow': 'auto'})
                ])
            ], width=7)
        ]),
    ], fluid=True, style={'height': '90vh', 'padding': '10px', 'width': '95vw'})


@app.callback([Output('graph', 'figure'), Output('table', 'columns'), Output('table', 'data')], [
    Input('run_button', 'n_clicks'),
    Input('v_l', 'value'), Input('v_b', 'value'), Input('v_h', 'value'), Input('v_t', 'value'), Input('cogx', 'value'),
    Input('cogy', 'value'), Input('cogz', 'value'), Input('p_l', 'value'), Input('p_w', 'value'), Input('p_h', 'value'),
    Input('t_min', 'value'), Input('t_max', 'value'), Input('n_t', 'value'), Input('d_min', 'value'),
    Input('d_max', 'value'), Input('n_d', 'value'), Input('water_depth', 'value'), Input('rho_water', 'value')
])
def run_diff(n_clicks, v_l, v_b, v_h, v_t, cogx, cogy, cogz, p_l, p_w, p_h, t_min, t_max, n_t, d_min, d_max, n_d,
             water_depth, rho_water):
    ctx = dash.callback_context
    if ctx.triggered:
        trigger_name = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_name == 'run_button':
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
            'd_max': d_max,
            'n_d': n_d,
            'water_depth': water_depth,
            'rho_water': rho_water,
        }
        RAOpd = initialize_calc(Values)
        columns = [{"name": i, "id": i} for i in RAOpd.columns]
        data = RAOpd.to_dict('records')
        figure = make_subplots(specs=[[{"secondary_y": True}]])
        figure.add_trace(go.Scatter(name='Surge', x=RAOpd["Period"].tolist(), y=RAOpd["Surge"].tolist()),
                         secondary_y=False, )
        figure.add_trace(go.Scatter(name='Sway', x=RAOpd["Period"].tolist(), y=RAOpd["Sway"].tolist()),
                         secondary_y=False, )
        figure.add_trace(go.Scatter(name='Heave', x=RAOpd["Period"].tolist(), y=RAOpd["Heave"].tolist()),
                         secondary_y=False, )
        figure.add_trace(go.Scatter(name='Roll', x=RAOpd["Period"].tolist(), y=RAOpd["Roll"].tolist()),
                         secondary_y=True, )
        figure.add_trace(go.Scatter(name='Pitch', x=RAOpd["Period"].tolist(), y=RAOpd["Pitch"].tolist()),
                         secondary_y=True, )
        figure.add_trace(go.Scatter(name='Yaw', x=RAOpd["Period"].tolist(), y=RAOpd["Yaw"].tolist()),
                         secondary_y=True, )
        figure.update_layout(title_text='Barge RAO', yaxis=dict(showexponent='all', exponentformat='e'))
        figure.update_xaxes(title_text='Period [s]')
        figure.update_yaxes(title_text='Translational RAOs [m/m]', secondary_y=False)
        figure.update_yaxes(title_text='Rotational RAOs [rad/m]', secondary_y=True)
        return figure, columns, data


def initialize_calc(Values):
    body = makemesh(Values)
    omega = np.linspace(float(Values["t_min"]), float(Values["t_max"]),
                        int(Values["n_t"]))
    counter = 0
    omegas = []
    RAO = []
    inputs = Values.copy()
    inputs["n_t"] = 1
    while counter < int(Values["n_t"]):
        omegas, RAO = calculation(inputs, omegas, counter, RAO, body, omega, Values)
        RAOpd = pd.DataFrame(RAO, columns=["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"])
        RAOpd.insert(0, "Period", omegas, True)
        counter += 1
    return RAOpd


def makemesh(Values):
    a = Values
    mesh = meshmaker(a["v_l"], a["v_b"], a["v_t"], a["p_l"], a["p_w"], a["p_h"])
    faces, vertices = mesh.barge()
    mesh = cpt.Mesh(vertices=vertices, faces=faces)
    body = cpt.FloatingBody(mesh=mesh, name="barge")
    return body


def calculation(inputs, omegas, counter, RAO, body, omega, Values):
    inputs["t_min"] = omega[counter]
    omegas.append(np.round(omega[counter], 2))
    RAO.append(EOM(body, inputs, show=False).solve())
    progress = (int(counter + 1) / int(Values["n_t"])) * 100
    return omegas, RAO


if __name__ == '__main__':
    app.run_server(debug=False)
    # app.run_server(mode='inline', debug=True)
