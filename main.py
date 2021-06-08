import numpy as np
import matplotlib.pyplot as plt
import re
from Solver.EOM import EOM
from kivy.lang import Builder
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.app import MDApp
from kmd_setup import mainapp
from kivymd.uix.textfield import MDTextField
from Solver.meshmaker import meshmaker
import capytaine as cpt
from kivy.properties import ObjectProperty
import pandas as pd
from kivy.clock import Clock

class RunDiff(MDBoxLayout):
    # progress = 0

    def __init__(self, **kwargs):
        super(RunDiff, self).__init__(**kwargs)
        self.progress = 0
        self.RAOpd = []
        self.calculation_trigger = Clock.create_trigger(self.calculation)

    def initialize_calc(self, *args):
        self.body = self.makemesh()
        self.omega = np.linspace(float(MainApp.Values["t_min"]), float(MainApp.Values["t_max"]),
                                 int(MainApp.Values["n_t"]))
        self.counter = 0
        self.omegas = []
        self.RAO = []
        self.inputs = MainApp.Values.copy()
        self.inputs["n_t"] = 1
        self.calculation_trigger()
        self.ids.textbox.size_hint = 1, None
        self.ids.textbox.height = (int(MainApp.Values["n_t"]) * 50)
        pd.options.display.precision = 2
        pd.set_option('display.colheader_justify', 'center')
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None
        pd.options.display.width = None

    def calculation(self, *args):
        self.inputs["t_min"] = self.omega[self.counter]
        self.omegas.append(self.omega[self.counter])
        self.RAO.append(EOM(self.body, self.inputs, show=False).solve())

        self.progress = (int(self.counter + 1) / int(MainApp.Values["n_t"])) * 100
        self.updbar()

        self.RAOpd = pd.DataFrame(self.RAO, columns=["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"])
        self.RAOpd.insert(0, "Period", self.omegas, True)
        self.RAOpd_string = self.RAOpd.to_string()
        self.updlabel()

        ax = self.RAOpd.plot(x='Period', y=['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw'],
                             secondary_y=['Roll', 'Pitch', 'Yaw'], grid=True)
        ax.set_ylabel('Translational RAO [m/m]')
        ax.right_ax.set_ylabel('Rotational RAO [rad/m]')
        ax.set_xlabel('Period [s]')
        ax.set_title('RAO')
        plt.savefig('plot.png', dpi= 500)
        plt.cla()
        plt.clf()
        plt.close()
        self.updplot()

        if float(self.inputs["t_min"]) < float(MainApp.Values["t_max"]):
            self.counter += 1
            self.calculation_trigger()

    def updbar(self, *args):
        self.ids.progbar.value = self.progress

    def updlabel(self, *args):
        # pdtabulate = lambda df: tabulate(df, headers='keys', tablefmt='psql')
        self.ids.results_label.text = self.RAOpd_string

    def updplot(self, *args):
        self.ids.plot.reload()

    def makemesh(self):
        a = MainApp.Values
        mesh = meshmaker(a["v_l"], a["v_b"], a["v_t"], a["p_l"], a["p_w"], a["p_h"])
        faces, vertices = mesh.barge()
        mesh = cpt.Mesh(vertices=vertices, faces=faces)
        body = cpt.FloatingBody(mesh=mesh, name="barge")
        return body


class MainApp(MDApp):
    Values = {
        'v_l': "122",
        'v_b': "32",
        'v_h': "8",
        'v_t': "4.875",
        'cogx': "0",
        'cogy': "0",
        'cogz': "15",
        'p_l': "4",
        'p_w': "4",
        'p_h': "4",
        't_min': "4",
        't_max': "30",
        'n_t': "10",
        'd_min': "0",
        'd_max': "0",
        'n_d': "1",
        'water_depth': "347.8",
        'rho_water': "1025",
    }
    AppInputs = Values.copy()
    run_diff = RunDiff()

    def on_text(self, name, value):
        self.Values[name] = value
        pass

    class ContentNavigationDrawer(MDBoxLayout):
        screen_manager = ObjectProperty()

    class FloatInput(MDTextField):
        pat = re.compile('[^0-9]')

        def insert_text(self, substring, from_undo=False):
            pat = self.pat
            if '.' in self.text:
                s = re.sub(pat, '', substring)
            else:
                s = '.'.join([re.sub(pat, '', s) for s in substring.split('.', 1)])
            return super(self.FloatInput, self).insert_text(s, from_undo=from_undo)

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Gray"
        self.theme_cls.primary_hue = "700"
        self.theme_cls.accent_palette = "LightBlue"
        build = Builder.load_string(mainapp)
        return build


if __name__ == '__main__':
    MainApp().run()
