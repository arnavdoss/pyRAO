import numpy as np
import matplotlib.pyplot as plt
import re
from EOM import EOM
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
from kivymd.app import MDApp
from kmd_setup import mainapp
from kivy.core.window import Window
from kivymd.uix.textfield import MDTextField
import threading
from meshmaker import meshmaker
import capytaine as cpt
from kivy.properties import ObjectProperty
import pandas as pd
# Window.size = (360, 740)


class pyRAOApp(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Gray"
        self.theme_cls.primary_hue = "700"
        self.theme_cls.accent_palette = "LightBlue"

        screen = Screen()
        screen.add_widget(Builder.load_string(mainapp))
        return screen

    def on_start(self):
        pass

    InputValues = {
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
        'n_w': "10",
        'd_min': "0",
        'd_max': "0",
        'n_d': "1",
        'water_depth': "10000",
        'rho_water': "1025",
    }
    RAO = [[1] * 6] * int(InputValues["n_w"])

    class ContentNavigationDrawer(BoxLayout):
        screen_manager = ObjectProperty()

    def makemesh(self):
        a = self.InputValues
        mesh = meshmaker(a["v_l"], a["v_b"], a["v_t"], a["p_l"], a["p_w"], a["p_h"])
        faces, vertices = mesh.barge()
        mesh = cpt.Mesh(vertices=vertices, faces=faces)
        body = cpt.FloatingBody(mesh=mesh, name="barge")
        return body

    def on_text(instance, name, value):
        pyRAOApp.InputValues[name] = value
        pass

    def runDiffraction(self):
        pyRAOApp.omega = np.linspace(float(pyRAOApp.InputValues["w_min"]), float(pyRAOApp.InputValues["w_max"]),
                                     int(pyRAOApp.InputValues["n_w"]))
        body = self.makemesh()
        self.RAO = []
        plotomega = []
        plotrao = {}
        plotrao[0] = []
        plotrao[1] = []
        plotrao[2] = []
        plotrao[3] = []
        plotrao[4] = []
        plotrao[5] = []
        for a, b in enumerate(self.omega):
            self.InputValues["w_min"] = b
            self.InputValues["w_max"] = b
            self.InputValues["n_w"] = 1
            self.RAO.append(EOM(body, self.InputValues, show=False).solve())
            plotomega.append(self.omega[a])
            plotrao[0].append(self.RAO[a][0])
            plotrao[1].append(self.RAO[a][1])
            plotrao[2].append(self.RAO[a][2])
            plotrao[3].append(self.RAO[a][3])
            plotrao[4].append(self.RAO[a][4])
            plotrao[5].append(self.RAO[a][5])
            self.plotRAO(plotomega, plotrao)
        self.RAOpd = pd.DataFrame([self.RAO], columns=["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"])

    def plotRAO(self, omega, RAO):
        print(omega)
        print(RAO)
        plt.figure()
        plt.plot(omega, RAO[0])
        plt.plot(omega, RAO[1])
        plt.plot(omega, RAO[2])
        plt.plot(omega, RAO[3])
        plt.plot(omega, RAO[4])
        plt.plot(omega, RAO[5])
        plt.legend(['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw'])
        plt.savefig('plot.png', dpi=600)
        plt.close()

    class FloatInput(MDTextField):
        pat = re.compile('[^0-9]')

        def insert_text(self, substring, from_undo=False):
            pat = self.pat
            if '.' in self.text:
                s = re.sub(pat, '', substring)
            else:
                s = '.'.join([re.sub(pat, '', s) for s in substring.split('.', 1)])
            return super(self.FloatInput, self).insert_text(s, from_undo=from_undo)


if __name__ == '__main__':
    pyRAOApp().run()
    #
    # v_l = 100  # Vessel length
    # v_b = 20  # Vessel beam
    # v_t = 2  # Vessel draft
    # v_h = 4  # Vessel height
    # p_l = 2  # Panel length
    # p_w = 2  # Panel width
    # p_h = 0.5  # Panel height
    # w_min = 0.01  # Min wave freq
    # w_max = 1  # Max wave freq
    # n_w = 10  # Number of wave frequencies
    # d_min = np.deg2rad(0)  # Min wave direction
    # d_max = 2 * np.pi  # Max wave direction
    # n_d = 1  # Number of wave directions
    # water_depth = np.infty  # Water depth
    # rho_water = 1025  # Density of water
    # cogx = 0  # Vessel longitudinal COG
    # cogy = 0  # Vessel transversal COG
    # cogz = 0  # Vessel vertical COG
    # grav_acc = 9.81  # Gravitational acceleration
    #
    #
    # class vessel:
    #     length = v_l
    #     beam = v_b
    #     draft = v_t
    #     height = v_h
    #     cogx = cogx
    #     cogy = cogy
    #     cogz = cogz
    #
    #
    # class panel:
    #     length = p_l
    #     width = p_w
    #     height = p_h
    #
    #
    # class diff_inputs:
    #     omega = np.linspace(w_min, w_max, n_w)
    #     wave_dir = np.linspace(d_min, d_max, n_d)
    #     water_depth = water_depth
    #     rho_water = rho_water
    #     grav_acc = grav_acc
    #
    # RAO = EOM.EOM(vessel, panel, diff_inputs, show=False).solve()
    #
    # rao11 = []
    # rao22 = []
    # rao33 = []
    # rao44 = []
    # rao55 = []
    # rao66 = []
    #
    # for a in range(n_w):
    #     rao11.append(RAO[a][0])
    #     rao22.append(RAO[a][1])
    #     rao33.append(RAO[a][2])
    #     rao44.append(RAO[a][3])
    #     rao55.append(RAO[a][4])
    #     rao66.append(RAO[a][5])
    #
    # plt.figure()
    # plt.subplot(231)
    # plt.plot(diff_inputs.omega, rao11)
    # plt.title('Surge')
    # plt.subplot(232)
    # plt.plot(diff_inputs.omega, rao22)
    # plt.title('Sway')
    # plt.subplot(233)
    # plt.plot(diff_inputs.omega, rao33)
    # plt.title('Heave')
    # plt.subplot(234)
    # plt.plot(diff_inputs.omega, rao44)
    # plt.title('Roll')
    # plt.subplot(235)
    # plt.plot(diff_inputs.omega, rao55)
    # plt.title('Pitch')
    # plt.subplot(236)
    # plt.plot(diff_inputs.omega, rao66)
    # plt.title('Yaw')
    # plt.show()
