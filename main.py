import numpy as np
import matplotlib.pyplot as plt
import re
from EOM import EOM
from kivy.lang import Builder
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.app import MDApp
from kmd_setup import mainapp
from kivymd.uix.textfield import MDTextField
from meshmaker import meshmaker
import capytaine as cpt
from kivy.properties import ObjectProperty
import pandas as pd
from kivy.clock import Clock

class updlbl(MDBoxLayout):

    def __init__(self, **kwargs):
        super(updlbl, self).__init__(**kwargs)
        # Clock.schedule_interval(self.upd, 1)

    def upd(self, *args):
        self.ids.updlbl.text = str("MainApp.run_diff.RAOpd")
        self.ids.rao_plot.reload()


class RunDiff(MDBoxLayout):

    # RAOpd = []
    progress = 0

    def __init__(self, **kwargs):
        super(RunDiff, self).__init__(**kwargs)
        self.RAOpd = []
        self.calculation_trigger = Clock.create_trigger(self.calculation)

    def initialize_calc(self, *args):
        self.body = self.makemesh()
        self.omega = np.linspace(float(MainApp.Values["w_min"]), float(MainApp.Values["w_max"]),
                                 int(MainApp.Values["n_w"]))
        self.counter = 0
        self.omegas = []
        self.RAO = []
        self.inputs = MainApp.Values.copy()
        self.inputs["n_w"] = 1
        self.calculation_trigger()
        self.ids.textbox.size_hint = 1, None
        self.ids.textbox.height = (int(MainApp.Values["n_w"])*45)

    def calculation(self, *args):
        self.inputs["w_min"] = self.omega[self.counter]
        self.omegas.append(self.omega[self.counter])
        self.RAO.append(EOM(self.body, self.inputs, show=False).solve())

        self.progress = (int(self.counter + 1) / int(MainApp.Values["n_w"])) * 100
        self.updbar()

        self.RAOpd = pd.DataFrame(self.RAO, columns=["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"])
        self.RAOpd.insert(0, "Omega", self.omegas, True)
        self.updlabel()

        ax = plt.gca()
        self.RAOpd.plot(kind='line', x='Omega', y='Surge', ax=ax)
        self.RAOpd.plot(kind='line', x='Omega', y='Sway', ax=ax)
        self.RAOpd.plot(kind='line', x='Omega', y='Heave', ax=ax)
        plt.savefig('plot_DOF123.png')
        plt.cla()

        ax = plt.gca()
        self.RAOpd.plot(kind='line', x='Omega', y='Roll', ax=ax)
        self.RAOpd.plot(kind='line', x='Omega', y='Pitch', ax=ax)
        self.RAOpd.plot(kind='line', x='Omega', y='Yaw', ax=ax)
        plt.savefig('plot_DOF456.png')
        plt.cla()
        self.updplot()

        if float(self.inputs["w_min"]) < float(MainApp.Values["w_max"]):
            self.counter += 1
            self.calculation_trigger()

    def updbar(self, *args):
        self.ids.progbar.value = self.progress

    def updlabel(self, *args):
        self.ids.results_label.text = str(self.RAOpd)

    def updplot(self, *args):
        self.ids.plot_DOF123.reload()
        self.ids.plot_DOF456.reload()

    def makemesh(self):
        a = MainApp.Values
        mesh = meshmaker(a["v_l"], a["v_b"], a["v_t"], a["p_l"], a["p_w"], a["p_h"])
        faces, vertices = mesh.barge()
        mesh = cpt.Mesh(vertices=vertices, faces=faces)
        body = cpt.FloatingBody(mesh=mesh, name="barge")
        return body

class MainApp(MDApp):
    Values = {
        'v_l': "100",
        'v_b': "20",
        'v_h': "6",
        'v_t': "3",
        'cogx': "50",
        'cogy': "0",
        'cogz': "2",
        'p_l': "2",
        'p_w': "2",
        'p_h': "1",
        'w_min': "0.05",
        'w_max': "1.2",
        'n_w': "5",
        'd_min': "0",
        'd_max': "0",
        'n_d': "1",
        'water_depth': "10000",
        'rho_water': "1025",
    }
    AppInputs = Values.copy()
    run_diff = RunDiff()
    results = updlbl()

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
