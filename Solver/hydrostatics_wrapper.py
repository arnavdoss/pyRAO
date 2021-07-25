import numpy as np
from Solver.meshmaker import meshmaker
from Solver import hydrostatics
from Solver.mesh import Mesh


def meshK(faces, vertices, cogx, cogy, cogz, rho_w, g):
    mesh = Mesh(vertices, faces)
    HS = hydrostatics.compute_hydrostatics(mesh, cog=[cogx, cogy, cogz], rho_water=rho_w, grav=g)
    HS_report = hydrostatics.get_hydrostatic_report(HS)
    CM = np.zeros((6, 6), float)
    for a in range(3):
        CM[a, a] = HS['disp_mass']
    IM = np.array([
        [HS['Ixx'], -HS['Ixy'], -HS['Ixz']],
        [-HS['Ixy'], HS['Iyy'], -HS['Iyz']],
        [-HS['Ixz'], -HS['Iyz'], HS['Izz']]])
    CK = np.zeros((6, 6), float)
    IK = HS['stiffness_matrix']
    for a in range(3):
        for b in range(3):
            CM[a+3, b+3] = IM[a, b]
            CK[a+2, b+2] = IK[a, b]
    return CM, CK, HS_report, HS


if __name__ == '__main__':
    length = 100
    beam = 30
    draft = 5
    height = 10
    xres = 4
    yres = 4
    zres = 4
    mesh = meshmaker(length, beam, height, draft, xres, yres, zres)
    faces, vertices = mesh.barge()
    # mesh = cpt.Mesh(vertices=vertices, faces=faces)
    # body = cpt.FloatingBody(mesh=mesh, name="barge")
    cogx = 0
    cogy = 0
    cogz = 15
    density = 1025
    g = 9.81
    meshK(faces, vertices, cogx, cogy, cogz, density, g)