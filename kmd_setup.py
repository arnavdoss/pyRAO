mainapp = """
#:import Transition kivy.uix.screenmanager.FadeTransition
#:import MDLabel kivymd.uix.label.MDLabel
#:import Clock kivy.clock.Clock
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
                                    icon: "chevron-left"
                                    on_press: screen_manager.current = "Input"
                                ToolbarButton:
                                    icon: "play-circle"
                                    on_press: screen_manager.current = "Results"
                                    on_release: Clock.schedule_once(app.process_button_click, 1)
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
                                    helper_text: "Length"                
                                InputText:                                    
                                    name: "v_b"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Breadth"
                                InputText:
                                    name: "v_h"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Height"
                                InputText:
                                    name: "v_t"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Draft"
                            InputBox:
                                InputHeaderIcon:
                                    icon: "ferry"
                                InputHeaderIcon:
                                    icon: "adjust"
                                InputText:
                                    name: "cogx"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "LCG"
                                InputText:
                                    name: "cogy"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "TCG"
                                InputText:
                                    name: "cogz"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "VCG"
                            InputBox:
                                InputHeaderIcon:
                                    icon: "ferry"
                                InputHeaderIcon:
                                    icon: "alpha-p-box-outline"
                                InputText:
                                    name: "p_l"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Length"
                                InputText:
                                    name: "p_w"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Width"
                                InputText:
                                    name: "p_h"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Height"
                                EmptyInputText:                           
                            InputBox:
                                InputHeaderIcon:
                                    icon: "waves"
                                InputHeaderIcon:
                                    icon: "alpha-f-circle-outline"
                                InputText:
                                    name: "w_min"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "min"
                                InputText:
                                    name: "w_max"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "max"
                                InputText:
                                    name: "n_w"                                    
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
                                    helper_text: "min"
                                InputText:
                                    name: "d_max"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "max"
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
                                    helper_text: "Depth"
                                InputText:
                                    name: "rho_water"                                    
                                    text: app.AppInputs[self.name]
                                    bind: app.on_text(self.name, self.text)
                                    helper_text: "Density"
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
                                ToolbarButton:
                                    icon: "play-circle"
                                    on_press: screen_manager.current = "Results"
                                    on_release: Clock.schedule_once(app.process_button_click, 0.1)
                                ToolbarButton:
                                    icon: "chevron-right"
                                    on_press: screen_manager.current = "Input"                               

                    ScrollView:
                        do_scroll_x: False
                        smooth_scroll_end: 10
                        always_overscroll: False             
                        MDStackLayout:
                            spacing: dp(10)
                            padding: dp(20)
                            size_hint: None, None
                            width: Window.width
                            Label:
                            updlbl:
                            
                                                                       
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
<updlbl>:
    orientation: 'vertical'
    size_hint: None, None
    width: Window.width
    adaptive_height: True
    Image:
        id: rao_plot
        source: 'plot.png'
        size: dp(500), dp(500)
    MDLabel:
        id: updlbl
                      
<InputBox@MDBoxLayout>:
    spacing: dp(10)
    padding: dp(5)                          
    size_hint: None, None
    size: dp(320), dp(50)
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
"""
