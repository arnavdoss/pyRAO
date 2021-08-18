import numpy as np
from Solver.meshmaker import meshmaker
from Solver import hydrostatics
from Solver.mesh import Mesh

def disp_calc(faces, vertices, cogx, cogy, cogz, rho_w, g):
    mesh = Mesh(vertices, faces)
    HS = hydrostatics.compute_hydrostatics(mesh, cog=[cogx, cogy, cogz], rho_water=rho_w, grav=g)
    return HS['disp_mass']

def COG_shift_cargo(cargo, disp, cogx, cogy, cogz):
    disp = disp * 1000
    mass = [float(cargo['c_mass'][a]) * 1000 for a in range(len(cargo['c_mass']))]
    c_x = [float(cargo['c_x'][a]) for a in range(len(cargo['c_x']))]
    c_y = [float(cargo['c_y'][a]) for a in range(len(cargo['c_y']))]
    c_z = [float(cargo['c_z'][a]) for a in range(len(cargo['c_z']))]
    cogx_global = []
    cogy_global = []
    cogz_global = []
    for a in range(len(mass)):
        cogx_global.append(mass[a] * c_x[a])
        cogy_global.append(mass[a] * c_y[a])
        cogz_global.append(mass[a] * c_z[a])
    cogx_tot = (sum(cogx_global) + ((disp - sum(mass)) * cogx)) / disp
    cogy_tot = (sum(cogy_global) + ((disp - sum(mass)) * cogy)) / disp
    cogz_tot = (sum(cogz_global) + ((disp - sum(mass)) * cogz)) / disp
    cog_cargo = [cogx_tot, cogy_tot, cogz_tot]
    return cog_cargo


def meshK(faces, vertices, cogx, cogy, cogz, rho_w, g, cargo):
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
            CM[a + 3, b + 3] = IM[a, b]
            CK[a + 2, b + 2] = IK[a, b]
    CM_cargo, cog_cargo = add_cargo(CM, HS['disp_mass'], cogx, cogy, cogz, cargo)
    return CM_cargo, CK, cog_cargo, HS_report, HS


def add_cargo(CM, disp, cogx, cogy, cogz, cargo):
    mass = [float(cargo['c_mass'][a])*1000 for a in range(len(cargo['c_mass']))]
    c_l = [float(cargo['c_l'][a]) for a in range(len(cargo['c_l']))]
    c_w = [float(cargo['c_w'][a]) for a in range(len(cargo['c_w']))]
    c_h = [float(cargo['c_h'][a]) for a in range(len(cargo['c_h']))]
    c_x = [float(cargo['c_x'][a]) for a in range(len(cargo['c_x']))]
    c_y = [float(cargo['c_y'][a]) for a in range(len(cargo['c_y']))]
    c_z = [float(cargo['c_z'][a]) for a in range(len(cargo['c_z']))]
    cogx_global = []
    cogy_global = []
    cogz_global = []
    for a in range(len(mass)):
        cogx_global.append(mass[a] * c_x[a])
        cogy_global.append(mass[a] * c_y[a])
        cogz_global.append(mass[a] * c_z[a])
    cogx_tot = (sum(cogx_global) + ((disp-sum(mass)) * cogx)) / disp
    cogy_tot = (sum(cogy_global) + ((disp-sum(mass)) * cogy)) / disp
    cogz_tot = (sum(cogz_global) + ((disp-sum(mass)) * cogz)) / disp
    cog_cargo = [cogx_tot, cogy_tot, cogz_tot]
    Ixx_global = []
    Iyy_global = []
    Izz_global = []
    for a in range(len(mass)):
        Ixx_loc = mass[a] * (c_w[a] ** 2 + c_h[a] ** 2) / 12
        Iyy_loc = mass[a] * (c_l[a] ** 2 + c_h[a] ** 2) / 12
        Izz_loc = mass[a] * (c_l[a] ** 2 + c_w[a] ** 2) / 12
        delta_x = c_x[a] - cogx_tot
        delta_y = c_y[a] - cogy_tot
        delta_z = c_z[a] - cogz_tot
        Ixx_global.append(Ixx_loc + (mass[a] * (delta_y ** 2 + delta_z ** 2) / 12))
        Iyy_global.append(Iyy_loc + (mass[a] * (delta_x ** 2 + delta_z ** 2) / 12))
        Izz_global.append(Izz_loc + (mass[a] * (delta_x ** 2 + delta_y ** 2) / 12))
    Ixx = sum(Ixx_global)
    Iyy = sum(Iyy_global)
    Izz = sum(Izz_global)
    CM[0][0] = CM[0][0] + sum(mass)
    CM[1][1] = CM[1][1] + sum(mass)
    CM[2][2] = CM[2][2] + sum(mass)
    CM[3][3] = CM[3][3] + Ixx
    CM[4][4] = CM[4][4] + Iyy
    CM[5][5] = CM[5][5] + Izz
    return CM, cog_cargo


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
    cargo = {'c_mass': [1000, 500], 'c_l': [50, 25], 'c_w': [20, 10], 'c_h': [10, 5], 'c_x': [50, 25], 'c_y': [0, 0],
             'c_z': [10, 5]}
    meshK(faces, vertices, cogx, cogy, cogz, density, 9.81, cargo)
