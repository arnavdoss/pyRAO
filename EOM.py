import meshmaker as mm
import capytaine as cpt
import numpy as np
import xarray as xr


class EOM:
    def __init__(self, body, inputs, show):
        self.body = body
        self.inputs = {k: np.float(v) for k, v in inputs.items()}
        self.show = show

    def solve(self):
        v_l = self.inputs["v_l"]
        v_b = self.inputs["v_b"]
        v_t = self.inputs["v_t"]
        v_h = self.inputs["v_h"]
        cogx = self.inputs["cogx"]
        cogy = self.inputs["cogy"]
        cogz = self.inputs["cogz"]
        KG = cogz + v_t
        p_l = self.inputs["p_l"]
        p_w = self.inputs["p_w"]
        p_h = self.inputs["p_h"]
        w_min = (2*np.pi)/self.inputs["t_min"]
        w_max = (2*np.pi)/self.inputs["t_max"]
        n_w = self.inputs["n_t"]
        d_min = np.deg2rad(self.inputs["d_min"])
        d_max = np.deg2rad(self.inputs["d_max"])
        n_d = self.inputs["n_d"]
        water_depth = self.inputs["water_depth"]
        rho_water = self.inputs["rho_water"]
        grav_acc = 9.81

        omega = np.linspace(w_min, w_max, int(n_w))
        wave_dir = np.linspace(d_min, d_max, int(n_d))
        Awl = v_l * v_b
        nabla = v_l * v_b * v_t
        cobz = v_t / 2
        mass = rho_water * nabla

        Mk = np.zeros((6, 6))
        Mk[0, 0] = mass
        Mk[1, 1] = mass
        Mk[2, 2] = mass
        Mk[3, 3] = (1 / 12) * (v_b ** 2 + v_h ** 2) * mass
        Mk[4, 4] = (1 / 12) * (v_h ** 2 + v_l ** 2) * mass
        Mk[5, 5] = (1 / 12) * (v_l ** 2 + v_b ** 2) * mass
        Mk[4, 5] = -(1 / 12) * (v_l * v_h) * mass
        Mk[5, 3] = Mk[4, 5]
        # Mk[5, 4] = -(1 / 12) * (v_b * v_h) * mass
        # Mk[4, 3] = Mk[5, 4]



        Ck = np.zeros((6, 6))
        Ck[2, 2] = Awl
        Ck[3, 3] = -(nabla * KG) + (nabla * cobz) + (1 / 12 * np.power(v_b, 3) * np.power(v_l, 3)) + (Awl * cogy ** 2)
        Ck[4, 4] = -(nabla * KG) + (nabla * cobz) + (1 / 12 * np.power(v_l, 3) * np.power(v_b, 3)) + (Awl * cogx ** 2)
        Ck[2, 3] = Awl * cogy
        Ck[3, 2] = Ck[2, 3]
        Ck[2, 4] = -Awl * cogx
        Ck[4, 2] = Ck[2, 4]
        Ck[3, 4] = -Awl * cogx * cogy
        Ck[4, 3] = Ck[3, 4]
        Ck[3, 5] = (nabla - nabla) * cogx
        Ck[4, 5] = (nabla - nabla) * cogy
        Ck = Ck * rho_water * grav_acc

        CM, CA, Fex = self.solvediff(self.body, v_l, v_b, v_t, p_l, p_w, p_h, omega, wave_dir, water_depth, cogx, cogy,
                                     cogz, self.show)
        RAO = self.solveeom(w_min, Mk, np.array(CM[w_min]), np.array(CA[w_min]), Ck, Fex[w_min])
        RAO = RAO.tolist()
        RAO = [item for sublist in RAO for item in sublist]
        return RAO
        # self.animate(omega.tolist()[0], body, RAO)
        # return RAO

    def solvediff(self, body, v_l, v_b, v_t, p_l, p_w, p_h, omega, wave_dir, water_depth, cogx, cogy, cogz, show):

        # body.add_all_rigid_body_dofs()
        axisx = cpt.Axis(vector=(1, 0, 0), point=(cogx, cogy, cogz))
        axisy = cpt.Axis(vector=(0, 1, 0), point=(cogx, cogy, cogz))
        axisz = cpt.Axis(vector=(0, 0, 1), point=(cogx, cogy, cogz))
        body.add_translation_dof(name="Surge")
        body.add_translation_dof(name="Sway")
        body.add_translation_dof(name="Heave")
        body.add_rotation_dof(axis=axisx, name="Roll", amplitude=1.0)
        body.add_rotation_dof(axis=axisy, name="Pitch", amplitude=1.0)
        body.add_rotation_dof(axis=axisz, name="Yaw", amplitude=1.0)

        if show:
            body.show()
        problems = xr.Dataset(coords={
            'omega': omega,
            'wave_direction': wave_dir,
            'radiating_dof': list(body.dofs),
            'water_depth': [water_depth]
        })
        dataset = cpt.BEMSolver().fill_dataset(problems, [body])
        CM = {}
        CA = {}
        F_diff = {}
        F_fk = {}
        Fex = {}
        for a, b in enumerate(omega.tolist()):
            CM[b] = dataset["added_mass"].values[a].tolist()
            CA[b] = dataset["radiation_damping"].values[a].tolist()
            F_diff[b] = dataset["diffraction_force"].values[a].tolist()
            F_fk[b] = dataset["Froude_Krylov_force"].values[a].tolist()
            Fex[b] = np.array(F_diff[b]) + np.array(F_fk[b])
        return CM, CA, Fex

    def solveeom(self, omega, Mk, CM, CA, Ck, Fex):
        # print(Fex)
        LHS = (-np.power(omega, 2) * (Mk + CM)) - (1j * omega * CA) + Ck
        RHS = Fex.transpose()
        RAO = np.linalg.pinv(LHS).dot(RHS)
        # try lstsq?
        RAO = np.absolute(RAO)
        return RAO

    def animate(self, omega, body, RAO):
        anim = body.animate(
            motion={"Surge": RAO[0.5][0], "Sway": RAO[0.5][1], "Heave": RAO[0.5][2], "Roll": RAO[0.5][3],
                    "Pitch": RAO[0.5][4], "Yaw": RAO[0.5][5]}, loop_duration=1.0)
        anim.run()
