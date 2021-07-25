# -*- coding: utf-8 -*-
import dash  # pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
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
import dash_vtk
from Solver.RollDamping import IkedaAdditionalDamping

# from jupyter_dash import JupyterDash

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN], title='pyRAO', update_title=None)
# app.css.append_css({'external_url': app.get_asset_url('stylesheet.css')})
# app = JupyterDash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

Aqua = '#00ADEF'
Navy = '#00306B'
Gray = '#EBE9E9'
Signal = '#DD1C1A'

tabs_styles = {
    'height': '40px',
}
tab_style = {
    'border': '2px',
    'border-top-left-radius': '10px',
    'border-top-right-radius': '10px',
    'border-bottom-left-radius': '10px',
    'border-bottom-right-radius': '10px',
    'padding': '6px',
    'backgroundColor': Gray,
    'fontWeight': 'bold',
    'width': '50%'
}

tab_selected_style = {
    'border': '2px solid #00ADEF',
    # 'border-top-left-radius': '10px',
    'border-radius': '10px',
    'backgroundColor': Gray,
    'padding': '6px',
    'fontWeight': 'bold',
    'width': '50%'
}

button_style = {
    # 'opacity': '0',
    'backgroundColor': 'white',
    'border': '0px solid #EBE9E9',
}

input_style = {
    'width': '90px',
    'textAlign': 'center'
}

radio_style = {
    'backgroundColor': '#e9ecef',
    'border-radius': '5px',
    'border': '1px solid #ced4da',
    'vertical-align': 'middle'
}

app.layout = dbc.Container(
    [
        dbc.Row([
            dcc.Interval(id='interval', n_intervals=0, interval=500),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col([
                                html.Img(
                                    src=app.get_asset_url('python-logo.svg'),
                                    alt="Python Logo", width="55dp"),
                            ], width=1, align="start"),
                            dbc.Col([
                                html.H2('pyRAO', style={'color': Aqua}),
                            ], width=3, align="start"),
                            dbc.Col([
                                html.H5('Open-source diffraction app with Capytaine, Meshmagick & Dash'),
                            ], width=8, align="start"),
                        ], style={'height': '50px'})
                    ]),
                    dbc.CardBody([
                        dbc.Col([
                            html.Div([
                                dcc.Tabs(id="tabs_object", value='tab-1', children=[
                                    dcc.Tab(label='Mesh', value='tab-1', style=tab_style,
                                            selected_style=tab_selected_style, children=[
                                            html.Div(id='mesh_viewer',
                                                     style={"width": "100%", "height": "calc(100vh - 265px)"},
                                                     children=[html.H1('Mesh viewer loading...')]),
                                        ]),
                                    dcc.Tab(label='RAO Plot', value='tab-2', style=tab_style,
                                            selected_style=tab_selected_style, children=[
                                            html.Div(id='bigger_wrapper_div', children=[
                                                html.Div(id='wrapper_div', children=[
                                                    dcc.Store(id='RAO_data'),
                                                    dcc.Store(id='Value_data'),
                                                    dcc.Graph(id='graph', style={'height': 'calc(100vh - 265px)'}),
                                                    # dcc.Graph(id='graph_FRAO', style={'height': '70vh'}),
                                                ]),
                                            ]),
                                        ]),
                                    dcc.Tab(label='Hydrostatics Report', value='tab-3', style=tab_style,
                                            selected_style=tab_selected_style,
                                            children=[html.Div(id='dbc_table',
                                                               style={'backgroundColor': 'white', 'overflow': 'auto',
                                                                      'height': 'calc(100vh - 265px)'})]),
                                    # dcc.Tab(label='Response Plot', value='tab-4', style=tab_style, disabled=False,
                                    #         selected_style=tab_selected_style, children=[
                                    #         dbc.Row([
                                    #             html.H1(' ')
                                    #         ]),
                                    #         dbc.Row(form=True, children=[
                                    #             dbc.Col([
                                    #                 dbc.FormGroup([
                                    #                     html.H6('Wave properties')
                                    #                 ])
                                    #             ], width=3),
                                    #             dbc.Col([
                                    #                 dbc.FormGroup([
                                    #                     dbc.Input(id='Hs', type='number', value=2, persistence=False,
                                    #                               bs_size='sm', persistence_type='local', min=0,
                                    #                               inputMode='numeric', style=input_style),
                                    #                     dbc.Label('Wave Height [m]', html_for='Hs', size='sm')
                                    #                 ]),
                                    #
                                    #             ], width=2),
                                    #             dbc.Col([
                                    #                 dbc.FormGroup([
                                    #                     dbc.Input(id='Tp', type='number', value=5, persistence=False,
                                    #                               bs_size='sm', persistence_type='local',
                                    #                               inputMode='numeric', min=0, disabled=True,
                                    #                               style=input_style),
                                    #                     dbc.Label('Peak period [s]', html_for='Tp', size='sm')
                                    #                 ]),
                                    #             ], width=2),
                                    #             dbc.Col([
                                    #                 dbc.FormGroup([
                                    #                     dbc.Input(id='gamma', type='number', value=3.3,
                                    #                               persistence=False,
                                    #                               bs_size='sm', persistence_type='local',
                                    #                               inputMode='numeric', min=0, disabled=True,
                                    #                               style=input_style),
                                    #                     dbc.Label('Gamma [s]', html_for='gamma', size='sm')
                                    #                 ]),
                                    #             ], width=2),
                                    #         ]),
                                    #         dbc.Row([
                                    #             dcc.Graph(
                                    #                 id='response_graph', style={'height': '70vh'}
                                    #             )
                                    #         ])
                                    #     ]),
                                ], style=tabs_styles),
                            ])
                        ], width=12)
                    ]),
                    dbc.CardFooter([
                        dbc.Row([
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.InputGroupAddon(
                                        html.Button(id='run_button', style=button_style, children=[
                                            html.Img(height='40px ', title='Begin undamped diffraction analysis',
                                                     style={'position': 'absolute', 'left': '0%'},
                                                     src=app.get_asset_url('start_icon.svg'))]), addon_type="prepend"
                                    ), dbc.Progress(id='progress_bar',
                                                    style={"height": "40px", "width": "95%", 'position': 'absolute',
                                                           'right': '0%'}),
                                ], style={'width': '100%'})
                            ], width=12)
                        ], style={'height': '40px'}),
                    ]),
                ]),
            ], width=7),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        'Inputs'
                    ], style={'height': '50px'}),
                    dbc.CardBody(style={'overflow': 'auto', 'height': 'calc(100vh - 135px)'}, children=[
                        dbc.FormGroup([
                            dbc.Label('Barge global dimensions', width=3, html_for='input_vessel'),
                            dbc.Col([
                                dbc.Input(id='v_l', type='number', value=122, persistence=False, bs_size='sm',
                                          persistence_type='local', inputMode='numeric', style=input_style),
                                dbc.Label('Length [m]', html_for='v_l', size='sm')
                            ], width=2),
                            dbc.Col([
                                dbc.Input(id='v_b', type='number', value=36.6, persistence=False, bs_size='sm',
                                          persistence_type='local', inputMode='numeric', style=input_style),
                                dbc.Label('Beam [m]', html_for='v_b', size='sm')
                            ], width=2),
                            dbc.Col([
                                dbc.Input(id='v_h', type='number', value=7.6, persistence=False, bs_size='sm',
                                          persistence_type='local', inputMode='numeric', style=input_style),
                                dbc.Label('Height [m]', html_for='v_h', size='sm')
                            ], width=2),
                            dbc.Col([
                                dbc.Input(id='v_t', type='number', value=3.625, persistence=False, bs_size='sm',
                                          persistence_type='local', inputMode='numeric', style=input_style),
                                dbc.Label('Draft [m]', html_for='v_t', size='sm')
                            ], width=2)
                        ], row=True, id='input_vessel'),
                        dbc.FormGroup([
                            dbc.Label('Barge COG', width=3, html_for='input_COG'),
                            dbc.Col([
                                dbc.Input(id='cogx', type='number', value=0, persistence=False, bs_size='sm',
                                          persistence_type='local', inputMode='numeric', disabled=False,
                                          style=input_style),
                                dbc.Label('LCG [m]  Midship', html_for='cogx', size='sm')
                            ], width=2),
                            dbc.Col([
                                dbc.Input(id='cogy', type='number', value=0, persistence=False, bs_size='sm',
                                          persistence_type='local', inputMode='numeric', disabled=False,
                                          style=input_style),
                                dbc.Label('TCG [m] Centerline', html_for='cogy', size='sm')
                            ], width=2),
                            dbc.Col([
                                dbc.Input(id='cogz', type='number', value=18.5, persistence=False, bs_size='sm',
                                          persistence_type='local', inputMode='numeric', style=input_style),
                                dbc.Label('VCG [m]   Baseline', html_for='cogz', size='sm')
                            ], width=2),
                        ], row=True, id='input_COG'),
                        dbc.FormGroup([
                            dbc.Label('Mesh panel dimensions', width=3, html_for='input_panel'),
                            dbc.Col([
                                dbc.Input(id='p_l', type='number', value=4, persistence=False, bs_size='sm',
                                          persistence_type='local', inputMode='numeric', style=input_style),
                                dbc.Label('Length [m]', html_for='p_l', size='sm')
                            ], width=2),
                            dbc.Col([
                                dbc.Input(id='p_w', type='number', value=4, persistence=False, bs_size='sm',
                                          persistence_type='local', inputMode='numeric', style=input_style),
                                dbc.Label('Width [m]', html_for='p_w', size='sm')
                            ], width=2),
                            dbc.Col([
                                dbc.Input(id='p_h', type='number', value=1, persistence=False, bs_size='sm',
                                          persistence_type='local', inputMode='numeric', style=input_style),
                                dbc.Label('Height [m]', html_for='p_h', size='sm')
                            ], width=2),
                        ], row=True, id='input_panel'),
                        dbc.FormGroup([
                            dbc.Label('Waves', width=3, html_for='input_wave'),
                            dbc.Col([
                                dbc.Input(id='d_min', type='number', value=60, persistence=False, bs_size='sm',
                                          persistence_type='local', inputMode='numeric', min=0, max=360,
                                          step=15, style=input_style),
                                dbc.Label('Direction [deg]', html_for='d_min', size='sm')
                            ], width=2),
                            dbc.Col([
                                dbc.Input(id='t_min', type='number', value=5, persistence=False, bs_size='sm',
                                          persistence_type='local', inputMode='numeric', min=1,
                                          style=input_style),
                                dbc.Label('Minimum period [s]', html_for='t_min', size='sm')
                            ], width=2),
                            dbc.Col([
                                dbc.Input(id='t_max', type='number', value=20, persistence=False, bs_size='sm',
                                          persistence_type='local', inputMode='numeric', min=1,
                                          style=input_style),
                                dbc.Label('Maximum period [s]', html_for='t_max', size='sm')
                            ], width=2),
                            dbc.Col([
                                dbc.Input(id='n_t', type='number', value=20, persistence=False, bs_size='sm',
                                          persistence_type='local', inputMode='numeric', min=1,
                                          style=input_style),
                                dbc.Label('No. of periods', html_for='n_t', size='sm')
                            ], width=2)
                        ], row=True, id='input_wave'),
                        dbc.FormGroup([
                            dbc.Label('Water properties', width=3, html_for='input_water'),
                            dbc.Col([
                                dbc.Input(id='water_depth', type='number', value=50, persistence=False,
                                          bs_size='sm',
                                          persistence_type='local', inputMode='numeric', style=input_style),
                                dbc.Label('Depth [m]', html_for='water_depth', size='sm')
                            ], width=2),
                            dbc.Col([
                                dbc.Input(id='rho_water', type='number', value=1025, persistence=False,
                                          bs_size='sm',
                                          persistence_type='local', inputMode='numeric', disabled=True,
                                          style=input_style),
                                dbc.Label('Density [kg/m^3]', html_for='rho_water', size='sm')
                            ], width=3),
                        ], row=True, id='input_water'),
                        dbc.FormGroup([
                            dbc.Label('Options', width=3, html_for='add_att'),
                            dbc.Col([
                                dbc.Checklist(options=[{'value': 'YES', 'label': 'Upload Mesh'}], id='run_upload',
                                              switch=True),
                            ], width=2, style={'textAlign': 'center'}),
                            dbc.Col([
                                dbc.Checklist(options=[{'value': 'YES', 'label': 'B44'}], id='run_damped', switch=True),
                            ], width=2, style={'textAlign': 'center'}),
                        ], row=True, id='add_att'),
                        dbc.Collapse(id='collapse_upload', is_open=False, children=[
                            dbc.FormGroup([
                                dbc.Card([
                                    dbc.CardHeader([
                                        dbc.Label('Upload Mesh'),
                                    ]),
                                    dbc.CardBody([
                                        dcc.Upload(
                                            id='upload-data',
                                            children=html.Div(['Drag and Drop or ', html.A(['Select Files'], style={
                                                "text-decoration": "underline"})]),
                                            style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                                   'borderWidth': '1px', 'borderStyle': 'dashed',
                                                   'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
                                            # Allow multiple files to be uploaded
                                            multiple=False
                                        ),
                                    ])
                                ]),
                            ])
                        ]),
                        dbc.Collapse(id='collapse_B44', is_open=False, children=[
                            dbc.FormGroup([
                                dbc.Card([
                                    dbc.CardHeader([
                                        dbc.Label('Additional roll damping'),
                                    ]),
                                    dbc.CardBody([
                                        dbc.FormGroup([
                                            dbc.Col([
                                                html.H1(' '),
                                                dbc.Input(id='Cm', type='number', value=0.98, persistence=False,
                                                          bs_size='sm',
                                                          persistence_type='local', inputMode='numeric',
                                                          style=input_style),
                                                dbc.Label('Midship Coefficient ', html_for='Cm', size='sm')
                                            ], width=6),
                                            dbc.Col([
                                                html.H1(' '),
                                                dbc.Input(id='pct_crit', type='number', value=5, persistence=False,
                                                          bs_size='sm', persistence_type='local', inputMode='numeric',
                                                          style=input_style, max=10),
                                                dbc.Label('[%] of critical damping', html_for='pct_crit', size='sm')
                                            ], width=6),
                                        ], row=True)
                                    ])
                                ]),
                            ])
                        ]),
                    ]),
                    dbc.CardFooter([
                        html.P('© 2021, Arnav Doss', style={'fontSize': '12px', 'textAlign': 'right'})
                    ], style={'height': '40px'})
                ]),
            ], width=5)
        ]),
    ], fluid=True, style={'padding-top': '15px'})


@app.callback([Output('run_button', 'children')], [Input('run_button', 'n_clicks'), Input('Value_data', 'data')])
def button_image(n, Values_in):
    Values = pd.read_json(Values_in)
    a = [html.Img(height='40vw ', title='Begin undamped diffraction analysis',
                  style={'position': 'absolute', 'left': '0%'},
                  src=app.get_asset_url('start_icon.svg'))]
    if n and float(Values['counter']) < float(Values['n_t']):
        return [dbc.Spinner(color='danger', size='lg',
                            spinner_style={'position': 'absolute', 'left': '0%', 'backgroundColor': '#E9ECEF'})]
    else:
        return a


@app.callback([Output('Value_data', 'data'), Output('RAO_data', 'data'), Output('tabs_object', 'value')], [
    Input('run_button', 'n_clicks'),
    Input('v_l', 'value'), Input('v_b', 'value'), Input('v_h', 'value'),
    Input('v_t', 'value'), Input('cogx', 'value'), Input('cogy', 'value'), Input('cogz', 'value'),
    Input('p_l', 'value'), Input('p_w', 'value'), Input('p_h', 'value'), Input('t_min', 'value'),
    Input('t_max', 'value'), Input('n_t', 'value'), Input('d_min', 'value'), Input('water_depth', 'value'),
    Input('rho_water', 'value')
])
def initialize_value(n1, v_l, v_b, v_h, v_t, cogx, cogy, cogz, p_l, p_w, p_h, t_min, t_max, n_t, d_min,
                     water_depth, rho_water):
    ctx = dash.callback_context
    if ctx.triggered:
        trigger_name = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_name == 'run_button':
        Values = makeValues(v_l, v_b, v_h, v_t, cogx, cogy, cogz, p_l, p_w, p_h, t_min, t_max, n_t, d_min, water_depth,
                            rho_water, 0, 0)
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
        global global_body, Mk, Ck, HS_lock
        global_body, Mk, Ck, HS_report, faces, vertices, faces2, vertices2, HS_lock = makemesh(Valuespd)
        return [Values_json, RAOs_json, 'tab-2']
    else:
        raise PreventUpdate


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


@app.callback([Output('wrapper_div', 'children'), Output('progress_bar', 'value'), Output('progress_bar', 'children'),
               Output('run_damped', 'value')],
              [Input('Value_data', 'data'), Input('RAO_data', 'data'), Input('interval', 'n_intervals'),
               Input('Cm', 'value'), Input('pct_crit', 'value')],
              [State('run_damped', 'value')])
def run_diff(Values_json, RAOpd_json, n, Cm, pct_crit, run_damped):
    Valuespd_in = pd.read_json(Values_json)
    RAOpd_in = pd.read_json(RAOpd_json)
    omega = np.linspace(float(Valuespd_in["t_min"]), float(Valuespd_in["t_max"]), int(Valuespd_in["n_t"]))
    count = Valuespd_in['counter']
    if float(count) <= float(Valuespd_in['n_t']) + 1:
        progress = np.rint(((float(count) + 1) / float(Valuespd_in['n_t'])) * 100)
        inputs = Valuespd_in.copy()
        inputs['n_t'] = 1
        inputs['t_min'] = omega[count].tolist()[0]
        if run_damped == ['YES'] and float(Valuespd_in['B44']) == 0:
            inputs['d_min'] = 90
        RAO_val, FRAO_val = EOM(global_body, Mk, Ck, inputs, show=False).solve()
        RAO_val.insert(0, omega[count].tolist()[0])
        RAOpd_in.loc[len(RAOpd_in)] = RAO_val
        Valuespd_in['counter'] = float(Valuespd_in['counter']) + 1
        if progress == 100 and run_damped == ['YES'] and float(Valuespd_in['B44']) == 0:
            run_damped_out = run_damped
            B44 = damptheroll(Valuespd_in, RAOpd_in, Cm, pct_crit)
            a = Valuespd_in.to_dict('records')[0]
            Values = makeValues(a['v_l'], a['v_b'], a['v_h'], a['v_t'], a['cogx'], a['cogy'], a['cogz'], a['p_l'],
                                a['p_w'], a['p_h'], a['t_min'], a['t_max'], a['n_t'], a['d_min'], a['water_depth'],
                                a['rho_water'], 0, B44)
            RAOs = {
                'Period': 0,
                'Surge': 0,
                'Sway': 0,
                'Heave': 0,
                'Roll': 0,
                'Pitch': 0,
                'Yaw': 0,
            }
            Valuespd_in = pd.DataFrame.from_records([Values])
            RAOpd_in = pd.DataFrame.from_records([RAOs])
            figure_RAO = create_empty_figure()
            figure_title = f"Calculating roll damping"
        else:
            B44 = float(Valuespd_in['B44'])
            run_damped_out = run_damped
            figure_RAO = create_figure(RAOpd_in)
            if float(Valuespd_in['B44']) == 0:
                if run_damped == ['YES']:
                    figure_title = f"Undamped barge RAO ({float(inputs['d_min'])} deg waves)"
                else:
                    figure_title = f"Undamped barge RAO ({float(inputs['d_min'])} deg waves)"
            else:
                figure_title = f"Roll damped barge RAO ({pct_crit}% of critical damping) at {float(inputs['d_min'])} deg waves"
        RAOpd_out = RAOpd_in.to_json()
        Values_out = Valuespd_in.to_json()
        return [[
            html.H5(figure_title, style={'height': '20px'}),
            dcc.Store(id='RAO_data', data=RAOpd_out),
            dcc.Store(id='Value_data', data=Values_out),
            dcc.Graph(id='graph', figure=figure_RAO, style={'height': 'calc(100vh - 295px)'})],
            progress, f"{progress} %" if progress >= 5 else "", run_damped_out]
    else:
        raise PreventUpdate


def damptheroll(Valuespd, RAOpd_in, Cm, pct_crit):
    # Valuespd_in = Valuespd.to_dict('records')[0]
    Troll = RAOpd_in.loc[RAOpd_in['Roll'] == max(RAOpd_in['Roll']), 'Period'].item()
    # damping = IkedaAdditionalDamping(Valuespd_in['v_l'], Valuespd_in['v_b'], Valuespd_in['v_t'],
    #                                  Valuespd_in['cogz'], HS_lock['transversal_metacentric_height'], Troll,
    #                                  HS_lock['disp_mass'], Cm)
    critical_damping = 2 * 9.81 * HS_lock['disp_mass'] * HS_lock['transversal_metacentric_height'] / (2 * np.pi / Troll)
    global B44_global
    B44_global = critical_damping * (pct_crit / 100)
    return B44_global


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
    figure.update_layout(yaxis=dict(showexponent='all', exponentformat='e'))
    figure.update_xaxes(title_text='Period [s]',
                        range=[sorted(RAOpd['Period'].tolist())[1], max(RAOpd['Period'].tolist())])
    figure.update_yaxes(title_text='Translational RAOs [m/m]', secondary_y=False)
    figure.update_yaxes(title_text='Rotational RAOs [deg/m]', secondary_y=True)
    figure.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return figure


def create_empty_figure():
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.update_layout(yaxis=dict(showexponent='all', exponentformat='e'))
    figure.update_yaxes(title_text='Translational RAOs [m/m]', secondary_y=False)
    figure.update_yaxes(title_text='Rotational RAOs [deg/m]', secondary_y=True)
    return figure


def makemesh(a):
    mesh = meshmaker(a["v_l"], a["v_b"], 0, a["v_t"], a["p_l"], a["p_w"], a["p_h"])
    faces, vertices = mesh.barge()
    mesh = cpt.Mesh(vertices=vertices, faces=faces)
    body = cpt.FloatingBody(mesh=mesh, name="barge")
    mesh2 = meshmaker(a["v_l"], a["v_b"], a["v_h"] - a["v_t"], a["v_t"], a["v_l"], a["v_b"], a["v_t"])
    faces2, vertices2 = mesh2.barge()
    Mk, Ck, HS_report, HS = meshK(faces2, vertices2, float(a['cogx']), float(a['cogy']), float(a['cogz']),
                                  float(a['rho_water']), 9.81)
    return body, Mk, Ck, HS_report, faces, vertices, faces2, vertices2, HS


def makeValues(v_l, v_b, v_h, v_t, cogx, cogy, cogz, p_l, p_w, p_h, t_min, t_max, n_t, d_min, water_depth, rho_water,
               counter, B44):
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
        'counter': counter,
        'B44': B44,
    }
    return Values


@app.callback([Output('dbc_table', 'children')], [
    Input('v_l', 'value'), Input('v_b', 'value'), Input('v_h', 'value'),
    Input('v_t', 'value'), Input('cogx', 'value'), Input('cogy', 'value'), Input('cogz', 'value'),
    Input('p_l', 'value'), Input('p_w', 'value'), Input('p_h', 'value'), Input('t_min', 'value'),
    Input('t_max', 'value'), Input('n_t', 'value'), Input('d_min', 'value'), Input('water_depth', 'value'),
    Input('rho_water', 'value')
])
def HSOut(v_l, v_b, v_h, v_t, cogx, cogy, cogz, p_l, p_w, p_h, t_min, t_max, n_t, d_min, water_depth, rho_water):
    Values = makeValues(v_l, v_b, v_h, v_t, cogx, cogy, cogz, p_l, p_w, p_h, t_min, t_max, n_t, d_min, water_depth,
                        rho_water, 0, 0)
    global_body, Mk, Ck, HS_report, faces, vertices, faces2, vertices2, HS = makemesh(Values)
    HS_report = HS_report.splitlines()
    output = [html.P('\n \n \n')]
    for a in range(len(HS_report)):
        curline = HS_report[a].replace('-', '').replace('\t', '').replace('**', '^').replace('>', '          ').split(
            '          ')
        if len(curline) == 1:
            curline.append(' ')
        output.append(curline)
    data = pd.DataFrame(output, columns=['Parameter', 'Value'])
    data.loc[len(data)] = ['Rxx [m]', np.round(np.sqrt(HS['Ixx'] / HS['disp_mass']), 3)]
    data.loc[len(data)] = ['Ryy [m]', np.round(np.sqrt(HS['Iyy'] / HS['disp_mass']), 3)]
    data.loc[len(data)] = ['Rzz [m]', np.round(np.sqrt(HS['Izz'] / HS['disp_mass']), 3)]
    HS_data = dbc.Table.from_dataframe(data, striped=True, borderless=True, hover=True, size='md',
                                       style={'font-size': '12px'})
    return [HS_data]


@app.callback([Output('mesh_viewer', 'children')], [
    Input('v_l', 'value'), Input('v_b', 'value'), Input('v_h', 'value'),
    Input('v_t', 'value'), Input('cogx', 'value'), Input('cogy', 'value'), Input('cogz', 'value'),
    Input('p_l', 'value'), Input('p_w', 'value'), Input('p_h', 'value'), Input('t_min', 'value'),
    Input('t_max', 'value'), Input('n_t', 'value'), Input('d_min', 'value'), Input('water_depth', 'value'),
    Input('rho_water', 'value')
])
def MESHOut(v_l, v_b, v_h, v_t, cogx, cogy, cogz, p_l, p_w, p_h, t_min, t_max, n_t, d_min, water_depth, rho_water):
    Values = makeValues(v_l, v_b, v_h, v_t, cogx, cogy, cogz, p_l, p_w, p_h, t_min, t_max, n_t, d_min, water_depth,
                        rho_water, 0, 0)
    global_body, Mk, Ck, HS_report, faces, vertices, faces2, vertices2, HS = makemesh(Values)
    for b in range(len(faces)):
        faces[b].insert(0, 4)
    for c in range(len(faces2)):
        faces2[c].insert(0, 4)
    content = dash_vtk.View(background=[1, 1, 1], children=[
        dash_vtk.GeometryRepresentation(
            property={'color': (0.5, 0.5, 0.5), 'edgeColor': (0, 0, 0), 'lighting': False, 'edgeVisibility': True},
            children=[
                dash_vtk.PolyData(
                    points=np.hstack(vertices2),
                    lines=np.hstack(faces2),
                ),
            ],
        ),
        dash_vtk.GeometryRepresentation(
            property={'color': Navy, 'edgeColor': (0, 0, 0), 'lighting': False, 'opacity': 0.5,
                      'ambientColor': (1, 1, 1), 'specularColor': (1, 1, 1), 'diffuseColor': (0, 0, 1)},
            children=[
                dash_vtk.PolyData(
                    points=np.hstack(vertices),
                    lines=np.hstack(faces),
                    polys=np.hstack(faces),
                    # children=[
                    #     dash_vtk.CellData([
                    #         dash_vtk.DataArray(
                    #             name='onCells',
                    #             values=[0, 1],
                    #         )
                    #     ]),
                    # ]
                ),
            ],
        ),
    ])
    return [[content]]


@app.callback(
    [Output("collapse_B44", "is_open")],
    [Input("run_damped", "value")],
    [State("collapse_B44", "is_open")],
)
def toggle_collapse(n, is_open):
    if n == ['YES']:
        return [True]
    else:
        return [False]


@app.callback(
    [Output("collapse_upload", "is_open")],
    [Input("run_upload", "value")],
    [State("collapse_upload", "is_open")],
)
def toggle_collapse(n, is_open):
    if n == ['YES']:
        return [True]
    else:
        return [False]


if __name__ == '__main__':
    app.run_server(debug=True)
