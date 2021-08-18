# -*- coding: utf-8 -*-
import dash  # pip install dash
import dash_core_components as dcc
import dash_html_components as html
import base64
import os
from dash.dependencies import Input, Output, State, ALL, MATCH
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import capytaine as cpt  # conda install -c conda-forge capytaine
import numpy as np
from Solver.EOM import EOM
from Solver.meshmaker import meshmaker
import dash_bootstrap_components as dbc  # conda install -c conda-forge dash-bootstrap-components
from dash.exceptions import PreventUpdate
from Solver.hydrostatics_wrapper import meshK, disp_calc, COG_shift_cargo
import dash_vtk

# from stl import mesh  # pip install numpy-stl

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN], title='pyRAO', update_title=None,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])
# LUMEN, SANDSTONE, YETI, CERULEAN

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
main_style = {
    'backgroundColor': '#e9ecef',
    'border-radius': '5px',
    'border': '1px solid #ced4da',
    'vertical-align': 'middle'
}
input_card_header_style = {
    'height': '50px',
    'textAlign': 'center',
}
input_card_body_style = {
    'height': '185px',
    'textAlign': 'left',
}
input_card_footer_style = {
    'height': '10px',
    'textAlign': 'center',
}


def create_empty_figure():
    figure = make_subplots(specs=[[{"secondary_y": True}]], horizontal_spacing=0, vertical_spacing=0)
    figure.add_trace(go.Scatter(name='Surge', x=[0, 0], y=[0, 0]), secondary_y=False, )
    figure.add_trace(go.Scatter(name='Sway', x=[0, 0], y=[0, 0]), secondary_y=False, )
    figure.add_trace(go.Scatter(name='Heave', x=[0, 0], y=[0, 0]), secondary_y=False, )
    figure.add_trace(go.Scatter(name='Roll', x=[0, 0], y=[0, 0]), secondary_y=True, )
    figure.add_trace(go.Scatter(name='Pitch', x=[0, 0], y=[0, 0]), secondary_y=True, )
    figure.add_trace(go.Scatter(name='Yaw', x=[0, 0], y=[0, 0]), secondary_y=True, )
    figure.update_layout(yaxis=dict(showexponent='all', exponentformat='e'))
    figure.update_xaxes(title_text='Period [s]')
    figure.update_yaxes(title_text='Translational RAOs [m/m]', secondary_y=False)
    figure.update_yaxes(title_text='Rotational RAOs [deg/m]', secondary_y=True)
    figure.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    figure.update_layout(title={'text': 'Motion RAOs', 'x': 0.5, 'xanchor': 'center'})
    figure.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    figure.update_layout(modebar={'orientation': 'h', 'bgcolor': 'white', 'add': 'hovercompare'})
    return figure


mesh_viewer = [
    dbc.Collapse([
        html.Div(id='mesh_viewer', style={"width": "100%", "height": "100%"})
    ], id='collapse_vtk_barge', is_open=True, style={"width": "100%", "height": "100%"}),
    dbc.Collapse([
        html.Div(id='mesh_viewer_upload', style={"width": "100%", "height": "100%"}, children=['TEST']),
    ], id='collapse_vtk_upload', is_open=False, style={"width": "100%", "height": "100%"}),
]
plot_viewer = [
    html.Div(id='wrapper_div', children=[
        dcc.Store(id='RAO_data'),
        dcc.Store(id='Value_data'),
        dcc.Graph(id='graph', figure=create_empty_figure(), style={'height': '100%'},
                  config={'displayModeBar': True}),
        # dcc.Graph(id='graph_FRAO', style={'height': '50vh'}),
    ], style={'height': '100%'}),
]
report_viewer = [html.Div(id='dbc_table', style={'backgroundColor': 'white'})]
inputs_mesh = [
    dbc.Collapse(id='collapse_mesh_upload', children=[
        dbc.CardBody([
            dcc.Upload(
                id='upload-mesh',
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
    dbc.Collapse(id='collapse_mesh_barge', is_open=True, children=[
        dbc.FormGroup([
            dbc.Col([
                dbc.Input(id='v_l', type='number', value=122, persistence=False,
                          bs_size='sm', persistence_type='local', inputMode='numeric',
                          style=input_style),
                dbc.FormText('Vessel Length [m]')
            ]),
            dbc.Col([
                dbc.Input(id='v_b', type='number', value=36.6, persistence=False,
                          bs_size='sm', persistence_type='local', inputMode='numeric',
                          style=input_style),
                dbc.FormText('Vessel Beam [m]')
            ]),
            dbc.Col([
                dbc.Input(id='v_h', type='number', value=7.6, persistence=False,
                          bs_size='sm', persistence_type='local', inputMode='numeric',
                          style=input_style),
                dbc.FormText('Vessel Height [m]')
            ]),
            dbc.Col([
                dbc.Input(id='p_l', type='number', value=4, persistence=False,
                          bs_size='sm',
                          persistence_type='local', inputMode='numeric',
                          style=input_style),
                dbc.FormText('Panel-X [m]')
            ]),
            dbc.Col([
                dbc.Input(id='p_w', type='number', value=4, persistence=False,
                          bs_size='sm',
                          persistence_type='local', inputMode='numeric',
                          style=input_style),
                dbc.FormText('Panel-Y [m]')
            ]),
            dbc.Col([
                dbc.Input(id='p_h', type='number', value=1, persistence=False,
                          bs_size='sm',
                          persistence_type='local', inputMode='numeric',
                          style=input_style),
                dbc.FormText('Panel-Z [m]')
            ]),
        ], row=True),
    ])
]
inputs_env = [
    dbc.FormGroup([
        dbc.Col([
            dbc.Input(id='t_min', type='number', value=5, persistence=False, bs_size='sm',
                      persistence_type='local', inputMode='numeric', min=1,
                      style=input_style),
            dbc.FormText('Minimum period [s]')
        ]),
        dbc.Col([
            dbc.Input(id='t_max', type='number', value=20, persistence=False, bs_size='sm',
                      persistence_type='local', inputMode='numeric', min=1,
                      style=input_style),
            dbc.FormText('Maximum period [s]')
        ]),
        dbc.Col([
            dbc.Input(id='n_t', type='number', value=20, persistence=False, bs_size='sm',
                      persistence_type='local', inputMode='numeric', min=1,
                      style=input_style),
            dbc.FormText('Number of periods')
        ]),
        dbc.Col([
            dbc.Input(id='d_min', type='number', value=60, persistence=False, bs_size='sm',
                      persistence_type='local', inputMode='numeric', min=0, max=360,
                      step=15, style=input_style),
            dbc.FormText('Wave Direction [deg]')
        ]),
        dbc.Col([
            dbc.Input(id='water_depth', type='number', value=50, persistence=False,
                      bs_size='sm',
                      persistence_type='local', inputMode='numeric', style=input_style),
            dbc.FormText('Water Depth [m]')
        ]),
        dbc.Col([
            dbc.Input(id='rho_water', type='number', value=1025, persistence=False,
                      bs_size='sm',
                      persistence_type='local', inputMode='numeric', disabled=True,
                      style=input_style),
            dbc.FormText('Density [kg/m^3]')
        ]),
    ], row=True)
]
inputs_vessel = [
    dbc.FormGroup([
        dbc.Col([
            dbc.Input(id='cogx', type='number', value=0, persistence=False,
                      bs_size='sm', persistence_type='local', inputMode='numeric',
                      disabled=False, style=input_style),
            dbc.FormText('LCG [m]  Midship 0')
        ]),
        dbc.Col([
            dbc.Input(id='cogy', type='number', value=0, persistence=False,
                      bs_size='sm', persistence_type='local', inputMode='numeric',
                      disabled=False, style=input_style),
            dbc.FormText('TCG [m] Centerline 0')
        ]),
        dbc.Col([
            dbc.Input(id='cogz', type='number', value=18.5, persistence=False,
                      bs_size='sm', persistence_type='local', inputMode='numeric',
                      style=input_style),
            dbc.FormText('VCG [m]   Baseline 0')
        ]),
        dbc.Col([
            dbc.Input(id='v_mass', type='number', value=17000, persistence=False,
                      bs_size='sm', persistence_type='local', inputMode='numeric',
                      style=input_style),
            dbc.FormText('Lightship mass [t]')
        ]),
    ], row=True),
]
inputs_B44 = [
    dbc.FormGroup([
        dbc.Col([
            dbc.Input(id='Cm', type='number', value=0.98, persistence=False, bs_size='sm', disabled=True,
                      persistence_type='local', inputMode='numeric', style=input_style, max=1, min=0, step=0.01),
            dbc.FormText('Cm [-]')
        ]),
        dbc.Col([
            dbc.Input(id='pct_crit', type='number', value=10, persistence=False, bs_size='sm', persistence_type='local',
                      inputMode='numeric', style=input_style, max=10, min=0, step=0.001, disabled=True),
            dbc.FormText('[%] of critical damping')
        ]),
    ], row=True)
]
info_badges = [
    dbc.Badge('©2021, Arnav Doss', color='success'),
    dbc.Badge('Draft: 0 m', id='badge_v_t', color='info'),
    dbc.Badge(['Displacement: 0 MT'], id='badge_disp', color='info'),
    dbc.Badge('Center of gravity: [0, 0, 0] m', id='badge_COG', color='info'),
    dbc.Badge('Center of buoyancy: [0, 0, 0] m', id='badge_COB', color='info'),
    dbc.Badge('Radii of Gyration: [0, 0, 0] m', id='badge_ROG', color='info'),
    dbc.Badge('GMT: 0 m', id='badge_GMT', color='info'),
]
main_header = [
    dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                html.Div([], style={'width': '5px'}),
                html.Img(src=app.get_asset_url('pyRAO-logo.svg'), width="40px"),
                html.Div([], style={'width': '10px'}),
                dbc.ButtonGroup([
                    dbc.Button('pyRAO', outline=True, color='info', active=True),
                    dbc.Button('↥', id='upload_info', outline=True, color='info'),
                    dbc.Button('↧', id='download_info', outline=True, color='info'),
                    dbc.Button('⊞', id='add_cargo', outline=True, color='info'),
                    dbc.Button('🗐', id='open_hs_report', outline=True, color='info'),
                    dbc.Button('▷', id='run_button', outline=True, color='danger'),
                ], style={'height': '40px'}),
                html.Div([], style={'width': '10px'}),
                dbc.Progress(id='progress_bar', style={"height": "40px", 'width': 'calc(100% - 325px)'}),
            ], no_gutters=False, style={'width': '100%'}, align='start'),
        ], style={'height': '60px'}),
    ], style={'width': '100%'})
]

app.layout = dbc.Container([
    dcc.Interval(id='interval', n_intervals=0, interval=1000),
    html.Div([
        dbc.Row([dbc.Col(main_header, style={'width': '100%'})], justify='center'),
        dbc.Row([dbc.Col(info_badges)], justify='center', align='start'),
        dbc.Row([], style={'height': '5px'}),
    ], style={'position': 'sticky', 'top': '0', 'z-index': '3000'}),
    dbc.Row([
        dbc.Col([
            dbc.Card([dbc.CardBody(mesh_viewer, style={'height': '100%'})], style={'height': '60vh'}),
        ], xs=12, sm=12, md=12, lg=6, xl=6),
        dbc.Col([dbc.Card([dbc.CardBody(plot_viewer, style={'height': '100%'})],
                          style={'height': '60vh'}), ], xs=12, sm=12, md=12, lg=6, xl=6),
    ], no_gutters=True),
    dbc.Row([
        # dbc.Alert('Error', id='alert1', dismissable=False, is_open=False, fade=False)
    ], style={'height': '5px'}),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    dbc.ButtonGroup([
                        dbc.Button('Barge', id='mesh_barge', active=True, outline=True, color='info',
                                   style={'width': '50%'}, size='sm'),
                        dbc.Button('Upload mesh', id='mesh_upload', outline=True, color='info',
                                   style={'width': '50%'}, size='sm', disabled=True),
                    ], style={'width': '100%'}),
                ], style=input_card_header_style),
                dbc.CardBody(inputs_mesh, style=input_card_body_style),
            ]),
        ], xs=12, sm=12, md=6, lg=4, xl=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.P('Environment')], style=input_card_header_style),
                dbc.CardBody(inputs_env, style=input_card_body_style),
            ]),
        ], xs=12, sm=12, md=6, lg=4, xl=3),
        dbc.Col([
            dbc.Card([
                dcc.Store(id='v_update_data'),
                dbc.CardHeader([html.P('Lightship parameters')], style=input_card_header_style),
                dbc.CardBody(inputs_vessel, style=input_card_body_style),
            ]),
        ], xs=12, sm=12, md=6, lg=4, xl=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    dbc.ButtonGroup([
                        dbc.Button('Undamped Vessel', id='vessel_undamped', active=True, outline=True, color='info',
                                   style={'width': '50%'}, size='sm'),
                        dbc.Button('Roll damped vessel', id='vessel_damped', outline=True, color='info',
                                   style={'width': '50%'},
                                   size='sm'),
                    ], style={'width': '100%'}),
                ], style=input_card_header_style),
                dbc.CardBody(inputs_B44, style=input_card_body_style),
            ]),
        ], xs=12, sm=12, md=6, lg=4, xl=3),
    ], no_gutters=True),
    dbc.Row([], style={'height': '5px'}),
    dcc.Store(id='cargo_data'),
    dbc.Row([], id='cargo_list', align='top', style={'width': '100%'}, no_gutters=True),
    dbc.Row([], style={'height': '5px'}),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.Collapse([
                    dbc.CardHeader(['Hydrostatics Report'], style=input_card_header_style),
                    dbc.CardBody(report_viewer),
                    dbc.CardFooter([], style=input_card_footer_style)
                ], id='collapse_hs_report'),
            ]),
        ]),
    ]),
], fluid=True)


def create_cargo(n):
    input_cargo_item = dbc.Col([
        dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col([html.P(f'Cargo Item {n}')]),
                    dbc.Col([
                        dbc.FormGroup([
                            dbc.Input(id={'type': 'c_mass', 'index': n}, type='number', value=0, persistence=False,
                                      bs_size='sm', persistence_type='local', inputMode='numeric',
                                      disabled=False, style=input_style),
                            html.P('Weight [t]'),
                        ], row=True)
                    ]),
                ], align='start', justify='between'),
            ], style=input_card_header_style),
            dbc.CardBody([
                dbc.FormGroup([
                    dbc.Col([
                        dbc.Input(id={'type': 'c_l', 'index': n}, type='number', value=0, persistence=False,
                                  bs_size='sm', persistence_type='local', inputMode='numeric',
                                  disabled=False, style=input_style),
                        dbc.FormText('Length [m]')
                    ]),
                    dbc.Col([
                        dbc.Input(id={'type': 'c_w', 'index': n}, type='number', value=0, persistence=False,
                                  bs_size='sm', persistence_type='local', inputMode='numeric',
                                  disabled=False, style=input_style),
                        dbc.FormText('Width TCG [m]')
                    ]),
                    dbc.Col([
                        dbc.Input(id={'type': 'c_h', 'index': n}, type='number', value=0, persistence=False,
                                  bs_size='sm', persistence_type='local', inputMode='numeric',
                                  style=input_style),
                        dbc.FormText('Height VCG [m]')
                    ]),
                    dbc.Col([
                        dbc.Input(id={'type': 'c_x', 'index': n}, type='number', value=0, persistence=False,
                                  bs_size='sm', persistence_type='local', inputMode='numeric',
                                  disabled=False, style=input_style),
                        dbc.FormText('Global LCG [m]')
                    ]),
                    dbc.Col([
                        dbc.Input(id={'type': 'c_y', 'index': n}, type='number', value=0, persistence=False,
                                  bs_size='sm', persistence_type='local', inputMode='numeric',
                                  disabled=False, style=input_style),
                        dbc.FormText('Global TCG [m]')
                    ]),
                    dbc.Col([
                        dbc.Input(id={'type': 'c_z', 'index': n}, type='number', value=0, persistence=False,
                                  bs_size='sm', persistence_type='local', inputMode='numeric',
                                  style=input_style),
                        dbc.FormText('Global VCG [m]')
                    ]),
                ], row=True)
            ])
        ])
    ], xs=12, sm=12, md=6, lg=4, xl=3)
    return input_cargo_item


@app.callback([Output('cargo_list', 'children')], [Input('add_cargo', 'n_clicks')], [State('cargo_list', 'children')])
def add_cargo_ui(n, cargo):
    if n > 0:
        print(len(cargo))
        cargo_item = create_cargo(n)
        cargo.append(cargo_item)
        return [cargo]


@app.callback(
    [Output('cargo_data', 'data')],
    [Input({'type': 'c_mass', 'index': ALL}, 'value'), Input({'type': 'c_l', 'index': ALL}, 'value'),
     Input({'type': 'c_w', 'index': ALL}, 'value'), Input({'type': 'c_h', 'index': ALL}, 'value'),
     Input({'type': 'c_x', 'index': ALL}, 'value'), Input({'type': 'c_y', 'index': ALL}, 'value'),
     Input({'type': 'c_z', 'index': ALL}, 'value')])
def add_cargo_data(c_mass, c_l, c_w, c_h, c_x, c_y, c_z):
    cargo = {'c_mass': c_mass, 'c_l': c_l, 'c_w': c_w, 'c_h': c_h, 'c_x': c_x, 'c_y': c_y, 'c_z': c_z}
    return [cargo]


@app.callback([Output('run_button', 'children')], [Input('run_button', 'n_clicks'), Input('Value_data', 'data')])
def button_image(n, Values_in):
    Values = pd.read_json(Values_in)
    if n and float(Values['counter']) < float(Values['n_t']):
        return [dbc.Spinner(color='danger', size='sm', debounce=500,
                            spinner_style={'backgroundColor': '#E9ECEF'})]
    else:
        return ['▷']


@app.callback(
    [Output('v_update_data', 'data')],
    [Input('v_mass', 'value'), Input('v_l', 'value'), Input('v_b', 'value'), Input('v_h', 'value'),
     Input('cogx', 'value'), Input('cogy', 'value'), Input('cogz', 'value'), Input('rho_water', 'value'),
     Input('cargo_data', 'data')]
)
def calculate_draft(v_mass, v_l, v_b, v_h, cogx, cogy, cogz, rho_water, cargo):
    draft_check = False
    disp_input = v_mass + sum(cargo['c_mass'])
    COG = COG_shift_cargo(cargo, disp_input, cogx, cogy, cogz)
    v_t = 0.5
    while draft_check is False:
        mesh = meshmaker(v_l, v_b, v_h - v_t, v_t, v_l, v_b, v_t)
        faces, vertices = mesh.barge()
        disp = disp_calc(faces, vertices, COG[0], COG[1], COG[2], rho_water, 9.81) / 1000
        disp_diff = (disp_input - disp) / disp_input
        if disp_diff >= 0.5:
            v_t = v_t + 1
        elif disp_diff >= 0.1:
            v_t = v_t + 0.1
        elif disp_diff >= 0.01:
            v_t = v_t + 0.01
        elif disp_diff > 0.001:
            v_t = v_t + 0.001
        elif disp_diff < 0.001:
            draft_check = True
    v_update_data = {'v_t': v_t, 'cogx': COG[0], 'cogy': COG[1], 'cogz': COG[2]}
    print(COG)
    return [v_update_data]


@app.callback([Output('Value_data', 'data'), Output('RAO_data', 'data')], [
    Input('run_button', 'n_clicks'),
    Input('v_l', 'value'), Input('v_b', 'value'), Input('v_h', 'value'),
    Input('v_update_data', 'data'),
    Input('p_l', 'value'), Input('p_w', 'value'), Input('p_h', 'value'), Input('t_min', 'value'),
    Input('t_max', 'value'), Input('n_t', 'value'), Input('d_min', 'value'), Input('water_depth', 'value'),
    Input('rho_water', 'value'), Input('cargo_data', 'data'),
])
def initialize_value(n1, v_l, v_b, v_h, v_update_data, p_l, p_w, p_h, t_min, t_max, n_t, d_min,
                     water_depth, rho_water, cargo):
    v_t = v_update_data['v_t']
    cogx = v_update_data['cogx']
    cogy = v_update_data['cogy']
    cogz = v_update_data['cogz']
    ctx = dash.callback_context
    if ctx.triggered:
        trigger_name = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_name == 'run_button':
        RAOs = {
            'Period': 0,
            'Surge': 0,
            'Sway': 0,
            'Heave': 0,
            'Roll': 0,
            'Pitch': 0,
            'Yaw': 0,
        }
        RAOpd = pd.DataFrame.from_records([RAOs])
        RAOs_json = RAOpd.to_json()

        Values_initial = makeValues(v_l, v_b, v_h, v_t, cogx, cogy, cogz, p_l, p_w, p_h, t_min, t_max, n_t, d_min,
                                    water_depth, rho_water, 0, 0)
        Valuespd_initial = pd.DataFrame.from_records([Values_initial])
        global global_body, Mk, Ck, HS_lock
        global_body, Mk, Ck, COG, HS_report, faces, vertices, faces2, vertices2, HS_lock = makemesh(Valuespd_initial,
                                                                                                    cargo)
        Values = makeValues(v_l, v_b, v_h, v_t, COG[0], COG[1], COG[2], p_l, p_w, p_h, t_min, t_max, n_t, d_min,
                            water_depth, rho_water, 0, 0)
        Valuespd = pd.DataFrame.from_records([Values])
        Values_json = Valuespd.to_json()
        return [Values_json, RAOs_json]
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


@app.callback([Output('wrapper_div', 'children'), Output('progress_bar', 'value'), Output('progress_bar', 'children')],
              [Input('Value_data', 'data'), Input('RAO_data', 'data'), Input('interval', 'n_intervals'),
               Input('Cm', 'value'), Input('pct_crit', 'value')],
              [State('vessel_damped', 'active')])
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
        if run_damped and float(Valuespd_in['B44']) == 0:
            inputs['d_min'] = 90
        print(inputs)
        RAO_val, FRAO_val = EOM(global_body, Mk, Ck, inputs, show=False).solve()
        RAO_val.insert(0, omega[count].tolist()[0])
        RAOpd_in.loc[len(RAOpd_in)] = RAO_val
        Valuespd_in['counter'] = float(Valuespd_in['counter']) + 1
        if progress == 100 and run_damped and float(Valuespd_in['B44']) == 0:
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
        else:
            figure_RAO = create_figure(RAOpd_in)
            if float(Valuespd_in['B44']) == 0:
                if run_damped:
                    figure_RAO.update_layout(
                        title={'text': str(f"Undamped barge RAO ({float(inputs['d_min'])} deg waves)"), 'x': 0.5,
                               'xanchor': 'center'})
                else:
                    figure_RAO.update_layout(
                        title={'text': str(f"Undamped barge RAO ({float(inputs['d_min'])} deg waves)"), 'x': 0.5,
                               'xanchor': 'center'})
            else:
                figure_RAO.update_layout(
                    title={'text': str(
                        f"Roll damped barge RAO ({pct_crit}% of critical damping) at {float(inputs['d_min'])} deg waves"),
                        'x': 0.5, 'xanchor': 'center'})
        RAOpd_out = RAOpd_in.to_json()
        Values_out = Valuespd_in.to_json()
        return [[
            dcc.Store(id='RAO_data', data=RAOpd_out),
            dcc.Store(id='Value_data', data=Values_out),
            dcc.Graph(id='graph', figure=figure_RAO, style={'height': '100%'}, config={'displayModeBar': True})],
            progress, f"{progress} %" if progress >= 5 else ""]
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
    figure.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    figure.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    figure.update_layout(modebar={'orientation': 'h', 'bgcolor': 'white', 'add': 'hovercompare'})
    return figure


def makemesh(Values, cargo):
    mesh = meshmaker(Values["v_l"], Values["v_b"], 0, Values["v_t"], Values["p_l"], Values["p_w"], Values["p_h"])
    faces, vertices = mesh.barge()
    mesh = cpt.Mesh(vertices=vertices, faces=faces)
    body = cpt.FloatingBody(mesh=mesh, name="barge")
    mesh2 = meshmaker(Values["v_l"], Values["v_b"], Values["v_h"] - Values["v_t"], Values["v_t"], Values["v_l"],
                      Values["v_b"], Values["v_t"])
    faces2, vertices2 = mesh2.barge()
    Mk, Ck, COG, HS_report, HS = meshK(faces2, vertices2, float(Values['cogx']), float(Values['cogy']),
                                       float(Values['cogz']), float(Values['rho_water']), 9.81, cargo)
    return body, Mk, Ck, COG, HS_report, faces, vertices, faces2, vertices2, HS


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


@app.callback([Output('dbc_table', 'children'), Output('badge_disp', 'children'), Output('badge_COG', 'children'),
               Output('badge_COB', 'children'), Output('badge_ROG', 'children'), Output('badge_GMT', 'children'),
               Output('badge_v_t', 'children')], [
                  Input('v_l', 'value'), Input('v_b', 'value'), Input('v_h', 'value'),
                  Input('v_update_data', 'data'),
                  Input('p_l', 'value'), Input('p_w', 'value'), Input('p_h', 'value'), Input('t_min', 'value'),
                  Input('t_max', 'value'), Input('n_t', 'value'), Input('d_min', 'value'),
                  Input('water_depth', 'value'), Input('rho_water', 'value'), Input('cargo_data', 'data'),
              ])
def HSOut(v_l, v_b, v_h, v_update_data, p_l, p_w, p_h, t_min, t_max, n_t, d_min, water_depth, rho_water, cargo):
    v_t = v_update_data['v_t']
    cogx = v_update_data['cogx']
    cogy = v_update_data['cogy']
    cogz = v_update_data['cogz']
    Values = makeValues(v_l, v_b, v_h, v_t, cogx, cogy, cogz, p_l, p_w, p_h, t_min, t_max, n_t, d_min, water_depth,
                        rho_water, 0, 0)
    global_body, Mk, Ck, COG, HS_report, faces, vertices, faces2, vertices2, HS = makemesh(Values, cargo)
    HS_report = HS_report.splitlines()
    output = [html.P('\n \n \n')]
    for a in range(len(HS_report)):
        curline = HS_report[a].replace('-', '').replace('\t', '').replace('**', '^').replace('>', '          ').split(
            '          ')
        if len(curline) == 1:
            curline.append(' ')
        output.append(curline)
    ROG = [
        ['Rxx [m]', np.round(np.sqrt(HS['Ixx'] / HS['disp_mass']), 3)],
        ['Ryy [m]', np.round(np.sqrt(HS['Iyy'] / HS['disp_mass']), 3)],
        ['Rzz [m]', np.round(np.sqrt(HS['Izz'] / HS['disp_mass']), 3)]
    ]
    COB = np.round(HS['buoyancy_center'], 3)
    data = pd.DataFrame(output, columns=['Parameter', 'Value'])
    data.loc[len(data)] = ROG[0]
    data.loc[len(data)] = ROG[1]
    data.loc[len(data)] = ROG[2]
    HS_data = dbc.Table.from_dataframe(data, striped=True, borderless=True, hover=True, size='md',
                                       style={'font-size': '12px'})
    return [
        HS_data,
        f"Displacement: {np.round(HS['disp_mass'] / 1000, 1)} MT",
        f"Center of gravity: [{np.round(cogx, 3)}, {np.round(cogy, 3)}, {np.round(cogz, 3)}] m",
        f"Center of buoyancy: [{COB[0]}, {COB[1]}, {COB[2]}] m",
        f"Radii of gyration: [{ROG[0][1]}, {ROG[1][1]}, {ROG[2][1]}] m",
        f"GMT: {np.round(HS['transversal_metacentric_height'], 3)} m",
        f"Draft: {np.round(HS['draught'], 3)} m",
    ]


@app.callback([Output('mesh_viewer', 'children')], [
    Input('v_l', 'value'), Input('v_b', 'value'), Input('v_h', 'value'),
    Input('v_update_data', 'data'),
    Input('p_l', 'value'), Input('p_w', 'value'), Input('p_h', 'value'), Input('t_min', 'value'),
    Input('t_max', 'value'), Input('n_t', 'value'), Input('d_min', 'value'), Input('water_depth', 'value'),
    Input('rho_water', 'value'), Input('cargo_data', 'data'),
])
def MESHOut(v_l, v_b, v_h, v_update_data, p_l, p_w, p_h, t_min, t_max, n_t, d_min, water_depth, rho_water,
            cargo):
    v_t = v_update_data['v_t']
    cogx = v_update_data['cogx']
    cogy = v_update_data['cogy']
    cogz = v_update_data['cogz']
    Values = makeValues(v_l, v_b, v_h, v_t, cogx, cogy, cogz, p_l, p_w, p_h, t_min, t_max, n_t, d_min, water_depth,
                        rho_water, 0, 0)
    global_body, Mk, Ck, COG, HS_report, faces, vertices, faces2, vertices2, HS = makemesh(Values, cargo)
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
                ),
            ],
        ),
        dash_vtk.GeometryRepresentation(
            property={'color': Navy, 'edgeColor': (0, 0, 0), 'lighting': False, 'opacity': 0.5,
                      'ambientColor': (1, 1, 1), 'specularColor': (1, 1, 1), 'diffuseColor': (1, 0, 0)},
            children=[
                dash_vtk.Algorithm(
                    vtkClass='vtkSphereSource',
                    state={
                        'center': [cogx, cogy, cogz],
                        'radius': 2,
                        'resolution': 50,
                        'skipInnerFaces': True,
                    }
                )
            ]
        )
    ])
    return [[content]]


def save_file(name, content):
    UPLOAD_DIRECTORY = "/assets/app_uploaded_files"
    if not os.path.exists(UPLOAD_DIRECTORY):
        os.makedirs(UPLOAD_DIRECTORY)
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


@app.callback([Output('mesh_viewer_upload', 'children')],
              [Input('upload-mesh', 'contents'), Input('v_update_data', 'data')],
              [State('upload-mesh', 'filename')])
def MESHOut_upload(uploaded_mesh, v_update_data, filename):
    draft = v_update_data['v_t']
    cogx = v_update_data['cogx']
    cogy = v_update_data['cogy']
    cogz = v_update_data['cogz']
    ctx = dash.callback_context
    if ctx.triggered:
        trigger_name = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_name == 'upload-mesh':
        save_file(filename, uploaded_mesh)
        imported_mesh = cpt.FloatingBody.from_file(app.get_asset_url(f'app_uploaded_files/{filename}'),
                                                   file_format='stl')
        imported_mesh = imported_mesh.translated_z(-float(draft))
        xOy_Plane = cpt.Plane(point=(0, 0, 0), normal=(0, 0, 1))
        # stl_mesh = imported_mesh.clipped(xOy_Plane)
        vertices = imported_mesh.mesh.vertices
        faces = imported_mesh.mesh.faces
        print(faces)
        for b in range(len(faces)):
            faces[b][0] = 3
        print(faces)
        # for b in range(len(faces)):
        #     faces[b].insert(0, 4)
        # stl_mesh.show()
    content = dash_vtk.View(background=[1, 1, 1], children=[
        dash_vtk.GeometryRepresentation(
            property={'color': Navy, 'edgeColor': (0, 0, 0), 'lighting': False, 'opacity': 0.5,
                      'ambientColor': (1, 1, 1), 'specularColor': (1, 1, 1), 'diffuseColor': (0, 0, 1)},
            children=[
                dash_vtk.PolyData(
                    points=np.hstack(vertices),
                    lines=np.hstack(faces),
                    polys=np.hstack(faces),
                ),
            ],
        ),
        # dash_vtk.GeometryRepresentation(
        #     property={'color': (0.5, 0.5, 0.5), 'edgeColor': (0, 0, 0), 'lighting': False, 'edgeVisibility': True},
        #     children=[
        #         dash_vtk.Reader(
        #             vtkClass='vtkSTLReader',
        #             parseAsText=uploaded_mesh
        #         ),
        #     ],
        # ),
        dash_vtk.GeometryRepresentation(
            property={'color': Navy, 'edgeColor': (0, 0, 0), 'lighting': False, 'opacity': 0.5,
                      'ambientColor': (1, 1, 1), 'specularColor': (1, 1, 1), 'diffuseColor': (1, 0, 0)},
            children=[
                dash_vtk.Algorithm(
                    vtkClass='vtkSphereSource',
                    state={
                        'center': [cogx, cogy, cogz],
                        'radius': 2,
                        'resolution': 50,
                        'skipInnerFaces': True,
                    }
                )
            ]
        )
    ])
    return [[content]]


@app.callback(
    [Output("collapse_hs_report", "is_open")],
    [Input("open_hs_report", "n_clicks")],
    [State("collapse_hs_report", "is_open")])
def toggle_collapse(n, is_open):
    if n:
        if is_open:
            return [False]
        else:
            return [True]


@app.callback(
    [Output("collapse_mesh_upload", "is_open"), Output("collapse_mesh_barge", "is_open"),
     Output("collapse_vtk_upload", "is_open"), Output("collapse_vtk_barge", "is_open"),
     Output("mesh_upload", "active"), Output("mesh_barge", "active")],
    [Input("mesh_upload", "n_clicks"), Input("mesh_barge", "n_clicks")],
    [State("collapse_mesh_upload", "is_open"), State("collapse_mesh_barge", "is_open")])
def toggle_collapse(n1, n2, upload, barge):
    ctx = dash.callback_context
    if ctx.triggered:
        trigger_name = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_name == 'mesh_upload':
        if upload:
            return [upload, barge, upload, barge, upload, barge]
        else:
            return [True, False, True, False, True, False]
    if trigger_name == 'mesh_barge':
        if barge:
            return [upload, barge, upload, barge, upload, barge]
        else:
            return [False, True, False, True, False, True]


@app.callback(
    [Output("Cm", "disabled"), Output("pct_crit", "disabled"),
     Output("vessel_undamped", "active"), Output("vessel_damped", "active")],
    [Input("vessel_undamped", "n_clicks"), Input("vessel_damped", "n_clicks")],
    [State("vessel_undamped", "active"), State("vessel_damped", "active")])
def toggle_collapse(n1, n2, undamped, damped):
    ctx = dash.callback_context
    if ctx.triggered:
        trigger_name = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_name == 'vessel_undamped':
        return [True, True, True, False]
    if trigger_name == 'vessel_damped':
        return [False, False, False, True]


if __name__ == '__main__':
    app.run_server(debug=False)
