import numpy as np

class meshmaker:

    def __init__(self, length, beam, height, draft, xres, yres, zres):
        self.length = length
        self.beam = beam
        self.height = height
        self.draft = draft
        self.xres = xres
        self.yres = yres
        self.zres = zres

    def barge(self):
        length = float(self.length)
        beam = float(self.beam)
        height = float(self.height)
        draft = float(self.draft)
        xres = float(self.xres)
        yres = float(self.yres)
        zres = float(self.zres)

        def makeface(face, locval, x, y, z):
            vertices = []
            faces = []
            if face == 'bottom':
                ax = x
                bx = -y
                c = min(z)
            elif face == 'SB':
                ax = -x
                bx = z
                c = min(y)
            elif face == 'PS':
                ax = x
                bx = z
                c = max(y)
            elif face == 'aft':
                ax = y
                bx = z
                c = min(x)
            elif face == 'fwd':
                ax = -y
                bx = z
                c = max(x)
            for a in ax:
                for b in bx:
                    if face == 'bottom':
                        vertices.append([a, b, c])
                    elif face == 'SB' or face == 'PS':
                        vertices.append([a, c, b])
                    elif face == 'aft' or face == 'fwd':
                        vertices.append([c, a, b])
            na = len(ax)
            nb = len(bx)
            faceloc = np.zeros((na, nb))
            for a in range(na):
                for b in range(nb):
                    faceloc[a, b] = locval
                    locval += 1
            for a in range(na-1):
                for b in range(nb-1):
                    faces.append([faceloc[a, b], faceloc[a+1, b], faceloc[a+1, b+1], faceloc[a, b+1]])
            return vertices, faces, locval

        # Make arrays for the dimensions
        xvals = np.round(np.linspace(-length/2, length/2, int(np.ceil(length/xres))+1), 3)
        yvals = np.round(np.linspace(-beam/2, beam/2, int(np.floor(beam/yres))+1), 3)
        zvals = np.round(np.linspace(height, -draft, int(np.floor(draft/zres))+1), 3)

        locval = 0
        vertices = []
        faces = []
        for a in ['bottom', 'SB', 'PS', 'aft', 'fwd']:
            vertice, face, locvalout = makeface(a, locval, xvals, yvals, zvals)
            locval = locvalout
            vertices.extend(vertice)
            faces.extend(face)
        return faces, vertices
