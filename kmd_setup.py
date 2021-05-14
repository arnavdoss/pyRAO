mainapp = """
#:import Transition kivy.uix.screenmanager.FadeTransition
#:import MDLabel kivymd.uix.label.MDLabel
Screen:
    NavigationLayout:
        # x: toolbar.height
        ScreenManager:
            id: screen_manager
            transition: Transition()
            Screen:
                name: "Input"
                ScreenLayout:
                    HeaderLayout:
                        MDBoxLayout:
                            orientation: "vertical"
                            MDBoxLayout:
                                size_hint: 1, None
                                height: 150
                            MDBoxLayout:                    
                                ToolbarButton:
                                    icon: "language-python"                                    
                                    user_font_size: "30sp"
                                    on_press: app.runDiffraction()
                                MDLabel:
                                    text: "Input"
                                    text_style: "H1"
                                    theme_text_color: "Primary"
                                    font_size: "20sp"
                                ToolbarButton:
                                    icon: "chevron-left"
                                    on_press: screen_manager.current = "Input"
                                ToolbarButton:
                                    icon: "settings"
                                ToolbarButton:
                                    icon: "chevron-right"
                                    on_press: screen_manager.current = "Results"

                    ScrollView:    
                        do_scroll_x: False
                        smooth_scroll_end: 10
                        always_overscroll: False
                        halign: "center"
                        MDStackLayout:                     
                            orientation: "lr-tb"
                            spacing: 10
                            padding: 20
                            size_hint_x: 1
                            adaptive_height: True                            
                            InputBox:
                                InputHeaderIcon:
                                    icon: "ferry"
                                InputText:
                                    name: "v_l"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Length"
                                    text: "100"                         
                                InputText:                                    
                                    name: "v_b"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Breadth"
                                    text: "20"
                                InputText:
                                    name: "v_h"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Height"
                                    text: "6"
                                InputText:
                                    name: "v_t"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Draft"
                                    text: "3"
                            InputBox:
                                InputHeaderIcon:
                                    icon: "ferry"
                                InputHeaderIcon:
                                    icon: "adjust"
                                InputText:
                                    name: "cogx"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "LCG"
                                    text: "50"
                                InputText:
                                    name: "cogy"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "TCG"
                                    text: "0"
                                InputText:
                                    name: "cogz"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "VCG"
                                    text: "2"
                            InputBox:
                                InputHeaderIcon:
                                    icon: "ferry"
                                InputHeaderIcon:
                                    icon: "alpha-p-box-outline"
                                InputText:
                                    name: "p_l"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Length"
                                    text: "2"
                                InputText:
                                    name: "p_w"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Width"
                                    text: "2"
                                InputText:
                                    name: "p_h"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Height"
                                    text: "1"
                                EmptyInputText:                           
                            InputBox:
                                InputHeaderIcon:
                                    icon: "waves"
                                InputHeaderIcon:
                                    icon: "alpha-f-circle-outline"
                                InputText:
                                    name: "w_min"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "min"
                                    text: "0.05"
                                InputText:
                                    name: "w_max"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "max"
                                    text: "0.10"
                                InputText:
                                    name: "n_w"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "n"
                                    text: "2"
                            InputBox:
                                InputHeaderIcon:
                                    icon: "waves"
                                InputHeaderIcon:
                                    icon: "alpha-d-circle-outline"
                                InputText:
                                    name: "d_min"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "min"
                                    text: "0"
                                InputText:
                                    name: "d_max"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "max"
                                    text: "0"
                                InputText:
                                    name: "n_d"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "n"
                                    text: "1"
                            InputBox:
                                InputHeaderIcon:
                                    icon: "waves"
                                InputHeaderIcon:
                                    icon: "water"
                                InputText:
                                    name: "water_depth"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Depth"
                                    text: "100"
                                InputText:
                                    name: "rho_water"
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Density"
                                    text: "1025"
                                EmptyInputText:
                                             
            Screen:
                name: "Results"
                ScreenLayout:
                    HeaderLayout:
                        MDBoxLayout:
                            orientation: "vertical"
                            MDBoxLayout:
                                size_hint: 1, None
                                height: 150
                            MDBoxLayout:                    
                                ToolbarButton:
                                    icon: "language-python"                                    
                                    user_font_size: "30sp"
                                MDLabel:
                                    text: "Results"
                                    text_style: "H1"
                                    theme_text_color: "Primary"
                                    font_size: "20sp"
                                ToolbarButton:
                                    icon: "chevron-left"
                                    on_press: screen_manager.current = "Input"
                                ToolbarButton:
                                    icon: "settings"
                                    
                                ToolbarButton:
                                    icon: "chevron-right"
                                    on_press: screen_manager.current = "Input"                               

                    ScrollView:    
                        do_scroll_x: False
                        smooth_scroll_end: 10
                        always_overscroll: False                    
                        MDBoxLayout:
                            orientation: "vertical"
                            Image:
                                source: 'plot.png'
                                # size_hint_x: 0.4
                                # allow_stretch: True                     
                            
                
                            
        # # on button, use #nav_drawer.set_state("open")      
        # MDNavigationDrawer:
        #     id: nav_drawer
        #     ContentNavigationDrawer:
        #         screen_manager: screen_manager
        #         nav_drawer: nav_drawer
        #         ScrollView:
        #             MDList:
        #                 OneLineListItem:
        #                     text: "Inputs"
        #                     on_press:
        #                         nav_drawer.set_state("close")
        #                         screen_manager.current = "Input"
        #                 OneLineListItem:
        #                     text: "Results"
        #                     on_press:
        #                         nav_drawer.set_state("close")
        #                         screen_manager.current = "Results"
                                
<InputBox@MDBoxLayout>:
    spacing: 10
    padding: 5                          
    size_hint: None, None
    size: 320, 50
    halign: "center"
    canvas.before:
        Color:
            rgba: 0, 0, 0, 0.2
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [(20, 20)]
<InputText@MDTextField>:
    color_mode: "accent"
    line_anim: True
    size_hint: None, None
    size: 50, 100
    helper_text_mode: "persistent"
    halign: "center"
    valign: "bottom"
    input_filter: "float"
    write_tab: False
    multiline: False
    on_text: app.FloatInput()
<InputHeaderIcon@MDIcon>:
    size_hint: None, 1
    width: 50
    theme_text_color: "Primary"
    halign: "center"
<EmptyInputText@MDLabel>:
    halign: "center"
    size_hint: None, None
    size: 50, 100
<ToolbarButton@MDIconButton>:
    user_font_size: "20sp"
    center_y: 0.1
    theme_text_color: "Primary"
    halign: "center"
<HeaderLayout@MDBoxLayout>:
    size_hint: 1, None
    height: 200
    elevation: 10
    canvas.before:
        Color:
            rgba: app.theme_cls.accent_color
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [(20, 20)]
<ScreenLayout@MDBoxLayout>:
    orientation: "vertical"
    size_hint: 1, 1
    valign: "top"
    md_bg_color: app.theme_cls.primary_color
    padding: 5
"""