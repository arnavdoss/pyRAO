mainapp = """
#:import Transition kivy.uix.screenmanager.FadeTransition
#:import MDLabel kivymd.uix.label.MDLabel
#:import Clock kivy.clock.Clock
#:import Thread threading.Thread
Screen:
    id: "MainScreen"
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
                                height: dp(50)
                            MDBoxLayout:                    
                                ToolbarButton:
                                    icon: "settings"
                                MDLabel:
                                    text: "Input"
                                    text_style: "H1"
                                    theme_text_color: "Primary"
                                    size: self.texture_size
                                ToolbarButton:
                                    icon: "chevron-right"
                                    on_press: screen_manager.current = "Results"

                    MyScroll:
                        MDStackLayout:                     
                            orientation: "lr-tb"
                            spacing: dp(10)
                            padding: dp(20)
                            size_hint_x: 1
                            adaptive_height: True                            
                            InputBox:
                                InputHeaderIcon:
                                    icon: "ferry"
                                InputText:
                                    name: "v_l"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Length [m]"                
                                InputText:                                    
                                    name: "v_b"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Breadth [m]"
                                InputText:
                                    name: "v_h"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Height [m]"
                                InputText:
                                    name: "v_t"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Draft [m]"
                            InputBox:
                                InputHeaderIcon:
                                    icon: "ferry"
                                InputHeaderIcon:
                                    icon: "adjust"
                                InputText:
                                    name: "cogx"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "LCG [m]"
                                InputText:
                                    name: "cogy"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "TCG [m]"
                                InputText:
                                    name: "cogz"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "VCG [m]"
                            InputBox:
                                InputHeaderIcon:
                                    icon: "ferry"
                                InputHeaderIcon:
                                    icon: "alpha-p-box-outline"
                                InputText:
                                    name: "p_l"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Length [m]"
                                InputText:
                                    name: "p_w"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Width [m]"
                                InputText:
                                    name: "p_h"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Height [m]"
                                EmptyInputText:                           
                            InputBox:
                                InputHeaderIcon:
                                    icon: "waves"
                                InputHeaderIcon:
                                    icon: "alpha-t-circle-outline"
                                InputText:
                                    name: "t_min"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "min [s]"
                                InputText:
                                    name: "t_max"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "max [s]"
                                InputText:
                                    name: "n_t"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "n"
                            InputBox:
                                InputHeaderIcon:
                                    icon: "waves"
                                InputHeaderIcon:
                                    icon: "alpha-d-circle-outline"
                                InputText:
                                    name: "d_min"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "min [deg]"
                                InputText:
                                    name: "d_max"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "max [deg]"
                                InputText:
                                    name: "n_d"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "n"
                            InputBox:
                                InputHeaderIcon:
                                    icon: "waves"
                                InputHeaderIcon:
                                    icon: "water"
                                InputText:
                                    name: "water_depth"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Depth [m]"
                                InputText:
                                    name: "rho_water"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Density [kg/m^3]"
                                EmptyInputText:
                    MDBoxLayout:
                        md_bg_color: app.theme_cls.primary_color
                        adaptive_height: True
                        BotToolbarButton:
                            icon: "home"
                            on_press: screen_manager.current = "Input"
                        BotToolbarButton:
                            icon: "pencil"
                            on_press: screen_manager.current = "Input"
                        BotToolbarButton:
                            icon: "table"
                            on_press: screen_manager.current = "Results"
                        BotToolbarButton:
                            icon: "help"
                            on_press: screen_manager.current = "Results" 

            Screen:
                name: "Results"
                ScreenLayout:
                    HeaderLayout:
                        MDBoxLayout:
                            orientation: "vertical"
                            MDBoxLayout:
                                size_hint: 1, None
                                height: dp(50)
                            MDBoxLayout:                    
                                ToolbarButton:
                                    icon: "settings"
                                MDLabel:
                                    text: "Results"
                                    text_style: "H1"
                                    theme_text_color: "Primary"
                                    size: self.texture_size                                 
                                ToolbarButton:
                                    icon: "chevron-left"
                                    on_press: screen_manager.current = "Input"                            

                    RunDiff:
                            

                    MDBoxLayout:
                        md_bg_color: app.theme_cls.primary_color
                        adaptive_height: True
                        BotToolbarButton:
                            icon: "home"
                            on_press: screen_manager.current = "Input"
                        BotToolbarButton:
                            icon: "pencil"
                            on_press: screen_manager.current = "Input"
                        BotToolbarButton:
                            icon: "table"
                            on_press: screen_manager.current = "Results"
                        BotToolbarButton:
                            icon: "help"
                            on_press: screen_manager.current = "Results" 
<RunDiff>:
    orientation: "vertical"
    spacing: dp(5)
    padding: dp(10)
    InputBox:        
        size_hint: 1, None
        height: dp(50)
        halign: "center"
        MDIconButton:
            id: RunDiff
            size_hint: None, 1
            width: dp(50)
            center_y: 0.1
            theme_text_color: "Primary"
            halign: "center"
            icon: "play-outline"
            on_press: root.initialize_calc()
        MDProgressBar:
            id: progbar
            color: app.theme_cls.accent_color
    MyScroll:
        MDStackLayout:        
            id: resultsbox             
            orientation: "lr-tb"
            spacing: dp(10)
            adaptive_height: True
            size_hint_x: 1
            InputBox:
                size: root.width*0.98, root.height
                Image:
                    id: plot
                    source: 'plot.png'
            MDBoxLayout:
                orientation: "vertical"
                id: textbox
                adaptive_height: True
                padding: dp(10)
                MDLabel:
                    id: results_label
                    # pos_hint: {'right':1, 'top':2}
            

<InputBox@MDBoxLayout>:
    spacing: dp(15)
    padding: dp(5)                          
    size_hint: None, None
    size: dp(340), dp(50)
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
    size: dp(50), dp(100)
    helper_text_mode: "persistent"
    halign: "center"
    valign: "bottom"
    input_filter: "float"
    write_tab: False
    multiline: False
    on_text: app.FloatInput()
<InputHeaderIcon@MDIcon>:
    size_hint: None, 1
    width: dp(50)
    theme_text_color: "Primary"
    halign: "center"
<EmptyInputText@MDLabel>:
    halign: "center"
    size_hint: None, None
    size: dp(50), dp(100)
<ToolbarButton@MDIconButton>:
    size_hint: None, 1
    width: dp(50)
    center_y: 0.1
    theme_text_color: "Primary"
    halign: "center"
<BotToolbarButton@MDIconButton>:
    size_hint: 0.25, None
    center_y: 0
    theme_text_color: "Primary"
    halign: "center"
<HeaderLayout@MDBoxLayout>:
    size_hint: 1, None
    height: dp(100)
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
    spacing: dp(5)
<MyScroll@ScrollView>:    
    do_scroll_x: False
    smooth_scroll_end: 10
    always_overscroll: False
    halign: "center"
"""
