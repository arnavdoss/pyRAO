import numpy as np
import matplotlib.pyplot as plt
import re
from EOM import EOM
from kivy.lang import Builder
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.screenmanager import Screen
from kivymd.app import MDApp
from kmd_setup import mainapp
from kivymd.uix.button import MDIconButton
from kivy.core.window import Window
from kivymd.uix.textfield import MDTextField
import threading, time
from meshmaker import meshmaker
import capytaine as cpt
from kivy.properties import ObjectProperty, StringProperty
import pandas as pd
import time
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivymd.uix.label import MDLabel
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.factory import Factory
import multiprocessing
from kivy.clock import mainthread, Clock


# Window.size = (360, 740)

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
    RAOpd = "No diffraction run yet"

    def process_button_click(self, arg):
        self.RAO = []
        self.body = self.makemesh()
        self.omega = np.linspace(float(self.Values["w_min"]), float(self.Values["w_max"]),
                                 int(self.Values["n_w"]))
        self.counter = 0
        omegas = []
        while self.counter <= int(self.Values["n_w"])-1:
            inputs = self.Values.copy()
            inputs["w_min"] = self.omega[self.counter]
            inputs["n_w"] = 1
            omegas.append(self.omega[self.counter])
            self.calculation(self.RAO, self.body, inputs)
            MainApp.RAOpd = pd.DataFrame(self.RAO, columns=["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"])
            MainApp.RAOpd.insert(0, "Omega", omegas, True)
            self.counter += 1
            print(MainApp.RAOpd)

    def calculation(self, RAO, body, inputs):
        RAO.append(EOM(body, inputs, show=False).solve())

    def makemesh(self):
        a = self.Values
        mesh = meshmaker(a["v_l"], a["v_b"], a["v_t"], a["p_l"], a["p_w"], a["p_h"])
        faces, vertices = mesh.barge()
        mesh = cpt.Mesh(vertices=vertices, faces=faces)
        body = cpt.FloatingBody(mesh=mesh, name="barge")
        return body

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


class updlbl(MDBoxLayout):

    def __init__(self, **kwargs):
        super(updlbl, self).__init__(**kwargs)
        Clock.schedule_interval(self.upd, 1)
        pass

    def upd(self, txt):
        self.ids.updlbl.text = str(MainApp.RAOpd)
        self.ids.rao_plot.reload()


if __name__ == '__main__':
    MainApp().run()
