import matplotlib.pyplot as plt
import numpy as np
import EOM
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.properties import ObjectProperty
from kivymd.theming import ThemableBehavior
from kivymd.uix.list import MDList
from kivy.uix.screenmanager import Screen, ScreenManager
from kivymd.app import MDApp
from kmd_setup import mainapp

if __name__ == '__main__':


    class DemoApp(MDApp):
        class ContentNavigationDrawer(BoxLayout, StackLayout):
            screen_manager = ObjectProperty()
            nav_drawer = ObjectProperty()

        class DrawerList(ThemableBehavior, MDList):
            pass

        class InputScreen(Screen):
            pass

        def build(self):
            screen = Screen()
            mainnav = Builder.load_string(mainapp)
            screen.add_widget(mainnav)

            return screen

        def on_start(self):
            pass


    DemoApp().run()

    v_l = 100  # Vessel length
    v_b = 20  # Vessel beam
    v_t = 2  # Vessel draft
    v_h = 4  # Vessel height
    p_l = 2  # Panel length
    p_w = 2  # Panel width
    p_h = 0.5  # Panel height
    w_min = 0.01  # Min wave freq
    w_max = 1  # Max wave freq
    n_w = 10  # Number of wave frequencies
    d_min = np.deg2rad(0)  # Min wave direction
    d_max = 2 * np.pi  # Max wave direction
    n_d = 1  # Number of wave directions
    water_depth = np.infty  # Water depth
    rho_water = 1025  # Density of water
    cogx = 0  # Vessel longitudinal COG
    cogy = 0  # Vessel transversal COG
    cogz = 0  # Vessel vertical COG
    grav_acc = 9.81  # Gravitational acceleration


    class vessel:
        length = v_l
        beam = v_b
        draft = v_t
        height = v_h
        cogx = cogx
        cogy = cogy
        cogz = cogz


    class panel:
        length = p_l
        width = p_w
        height = p_h


    class diff_inputs:
        omega = np.linspace(w_min, w_max, n_w)
        wave_dir = np.linspace(d_min, d_max, n_d)
        water_depth = water_depth
        rho_water = rho_water
        grav_acc = grav_acc

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
