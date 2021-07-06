import numpy as np
import capytaine as cpt
from Solver.meshmaker import meshmaker
from meshmagick import hydrostatics

def meshK(body, cogx, cogy, cogz, mass, rho_w, g):
    print(body)
    print(mass)
    a = hydrostatics.Hydrostatics.hydrostatic_stiffness_matrix

if __name__ == '__main__':
    length = 100
    beam = 30
    draft = 3
    xres = 4
    yres = 4
    zres = 2
    mesh = meshmaker(length, beam, draft, xres, yres, zres)
    faces, vertices = mesh.barge()
    mesh = cpt.Mesh(vertices=vertices, faces=faces)
    body = cpt.FloatingBody(mesh=mesh, name="barge")
    cogx = 0
    cogy = 0
    cogz = 0
    density = 1023
    mass = density/1000 * (length * beam * draft)
    g = 9.81
    meshK(body, cogx, cogy, cogz, mass, density, g)
