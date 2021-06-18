import numpy as np
class meshviewer:

    def __init__(self, length, beam, draft, xres, yres, zres):
        self.length = length
        self.beam = beam
        self.draft = draft
        self.xres = xres
        self.yres = yres
        self.zres = zres

    def meshfig(self):
        length = float(self.length)
        beam = float(self.beam)
        draft = float(self.draft)
        xres = float(self.xres)
        yres = float(self.yres)
        zres = float(self.zres)