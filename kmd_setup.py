mainapp = """
Screen:
    MDBottomAppBar:
        MDToolbar:            
            id: toolbar
            type: "bottom"
            icon: "play"
            title: "MDNavigationDrawer"
            left_action_items: [["menu", lambda x: nav_drawer.set_state("open")]]
            mode: "end"

    NavigationLayout:
        x: toolbar.height
        ScreenManager:
            id: screen_manager
            InputScreen:               
                    
            Screen:
                name: "Results"
                MDLabel:
                    text: "Results"
                    halign: "center"

        MDNavigationDrawer:
            id: nav_drawer
            ContentNavigationDrawer:
                screen_manager: screen_manager
                nav_drawer: nav_drawer
                ScrollView:
                    MDList:
                        OneLineListItem:
                            text: "InputScreen"
                            on_press:
                                nav_drawer.set_state("close")
                                screen_manager.current = "Input"
                        OneLineListItem:
                            text: "ResultsScreen"
                            on_press:
                                nav_drawer.set_state("close")
                                screen_manager.current = "Results"
<InputScreen>:
    name: "Input"
"""