import numpy as np
import matplotlib.pyplot as plt

# Ref. [1] Naval_Empirical calculation of roll damping for shipsBarges (\\twd-fs01\data\Library\20 THEORY AND LITERATURE\00 Literature\Roll Damping\Literature\Company's database)
# Ref. [2] 20121108-OCTOPUS-Theoretical-Manual_ABB (B:\ICT\Marine\Octopus\V6.4.14)


class IkedaAdditionalDamping:

	# TODO: add all values needed for reporting

	def __init__(self, Vessel_length, Vessel_width, Vessel_draft, VCG, GMX, Troll, displacement, Cm):
		# Input values
		self.water_density = 1.025  			# [t/m3] Water
		self.g = 9.81  							# [m/s2] Acceleration due to gravity
		self.Vessel_length = Vessel_length  	# [m] length of vessel
		self.Vessel_width = Vessel_width  		# [m]
		self.Vessel_draft = Vessel_draft  		# [m] draft of vessel
		self.VCG = VCG  						# [m] Vertical centre of gravity
		self.GMX = GMX  						# [m] Transverse GM
		self.Troll = Troll  					# [s] Vessel roll period
		self.displacement = displacement		# [t] Vessel displacement
		self.OG = Vessel_draft - VCG  			# [m] Vertical distance from still water level to COG
		self.H0 = Vessel_width/Vessel_draft/2  	# [-] Half beam to draft ratio
		self.wave_freq = 2*np.pi/Troll			# [rad/s] Wave frequency
		self.Cm = Cm							# [-] Midship section coefficient
		self.BilgeKeel_breadth = None  			# MKB [m] breadth of bilge keel, if present
		self.BilgeKeel_length = None  			# MKB [m] length of bilge keel, if present
		# Calculated values
		self.B44BK = None
		self.Eddy_term = None  					# [kgm2]
		self.pct_crit = None  					# Percentage of critical damping applied at 0 speed
		self.PotentialDamp0kt = None  			# Potential damping at roll period at 0 speed
		self.B44e0 = None						# Eddy making damping at zero speed

		self.crit_damping = self.Calc_critical_damping()  # [kgm2/s]
		self.Calc_Eddy_term()

	def __str__(self):
		return "Vessel_length: {} m,\nVessel_width: {} m,\nVessel_draft: {} m,\nVCG: {} m,\nGMX: {} m,\nTroll: {} s,\ndisplacement: {} t,\nOG: {} m,\nH0: {},\ncrit_damping: {} kgm2/s,\nwave_freq: {} rad/s,\nEddy_term: {} kgm2".format(self.Vessel_length, self.Vessel_width, self.Vessel_draft, self.VCG, self.GMX, self.Troll, self.displacement, self.OG, self.H0, self.crit_damping, self.wave_freq, self.Eddy_term)

	def initBilgeKeel(self, BilgeKeel_breadth, BilgeKeel_length):
		self.BilgeKeel_breadth = BilgeKeel_breadth
		self.BilgeKeel_length = BilgeKeel_length

	def Calc_critical_damping(self):
		return 2*self.g*self.displacement*self.GMX/(2*np.pi/self.Troll) # [kgm2/s]

	def Calc_Eddy_term(self):
		A = 2/np.pi*self.water_density*self.Vessel_length*self.Vessel_draft**4 # [kgm2]
		B = self.H0**2 + 1 - self.OG/self.Vessel_draft # [-]
		C = (1 - self.OG/self.Vessel_draft)**2 + self.H0**2 # [-]
		self.Eddy_term = A*B*C # [kgm2]

	def Roll_amplitude(self, Pct_crit):
		self.pct_crit = Pct_crit
		damping = (Pct_crit/100)*self.crit_damping # [kgm2/s]
		roll_amp = (damping/(self.wave_freq*self.Eddy_term))*180/np.pi # [deg]
		sig_roll_amp = 1.86*roll_amp # [deg]
		return sig_roll_amp

	def B44(self, Pct_crit):
		self.pct_crit = Pct_crit
		return (Pct_crit/100)*self.crit_damping

	#MKB new function
	def BilgeKeel_damping(self, significant_R0):
		R0 = significant_R0/1.86*0.0174533
		A = self.Cm*self.Vessel_width*self.Vessel_draft 												# sectional area
		sigma = A/(self.Vessel_width*self.Vessel_draft)													# Ref. [2] (Equation 5.2-33) and discussion with James
		if self.H0 > 1:			
			R_b = self.Vessel_draft
		elif self.H0 < 1: 	
			R_b = self.Vessel_width/2
		R_b = min(R_b, 2 * self.Vessel_draft * np.sqrt(self.H0*( sigma - 1)/(np.pi-4)))					# Ref. [1] (59); or Ref. [2] (5.2-35)

		r_cb_1 = self.H0-0.293* R_b / self.Vessel_draft			
		r_cb_2 =  1 - self.OG / self.Vessel_draft - 0.293 * R_b / self.Vessel_draft
		r_cb = self.Vessel_draft*(r_cb_1**2 + r_cb_2**2)**(1/2)		# bilge keel mean distance from roll axis Ref. [1] (60)
		
		f = 1 + 0.3*np.exp(-160*(1-sigma))																# Ref. [1] (49)
		Cd = 22.5*self.BilgeKeel_breadth/(np.pi*r_cb* R0 * f)+2.4 										# Ref. [1] (48)
		B_BKN = 8/(3*np.pi)*self.water_density*r_cb**3*self.BilgeKeel_breadth*self.wave_freq* R0 *f**2*Cd		# Ref. [1] (47)
		
		m1 = R_b/self.Vessel_draft 																		# Ref. [1] (53)
		m2 = self.OG/self.Vessel_draft 																	# Ref. [1] (53)
		m3 = 1-m1-m2 																					# Ref. [1] (53)
		m4 = self.H0-m1 																				# Ref. [1] (53)
		m5 = (0.414*self.H0+0.0651*m1**2-(0.382*self.H0+0.0106)*m1)/((self.H0-0.215*m1)*(1-0.215*m1)) 	# Ref. [1] (54)
		m6 = (0.414*self.H0+0.0651*m1**2-(0.382+0.0106*self.H0)*m1)/((self.H0-0.215*m1)*(1-0.215*m1)) 	# Ref. [1] (55)
		S0 = 0.3*np.pi*f*r_cb* R0 + 1.95*self.BilgeKeel_breadth											# Ref. [1] (58)
		if S0>0.25*np.pi*R_b:
			m7 = S0/self.Vessel_draft-0.25*np.pi*m1														# Ref. [1] (56)
			m8 = m7+0.414*m1 																			# Ref. [1] (57)
		else:
			m7 = 0																						# Ref. [1] (56)
			m8 = m7 + np.sqrt(2)*(1-np.cos(S0/R_b))*m1														# Ref. [1] (57)

		A2 = (m3+m4)*m8-m7**2																			# Ref. [1] (51)
		B2 = (m4**3)/(3*(self.H0-0.215*m1))+((1-m1)**2)*(2*m3-m2)/(6*(1-0.215*m1))+(m3*m5+m4*m6)*m1 		# Ref. [1] (52)

		B_BKH_1 = 4/(3*np.pi)*self.water_density*(r_cb**2)*(self.Vessel_draft**2)*self.wave_freq*R0*(f**2)
		B_BKH_2 = -1*(-22.5*self.BilgeKeel_breadth/(np.pi*r_cb*f* R0) - 1.2)*A2 + 1.2* B2
		B_BKH = B_BKH_1*B_BKH_2																			# Ref. [1] (50)

		B_BK = (B_BKN + B_BKH)*self.BilgeKeel_length													# Ref. [1] (46)
		
		self.B44BK = B_BK
		return B_BK

	def ForwardSpeedEddy(self, Pct_crit, speed):
		self.pct_crit = Pct_crit
		eddy0knots = (Pct_crit/100)*self.crit_damping
		factor = ((0.04*self.wave_freq*self.Vessel_length/speed)**2)/((0.04*self.wave_freq*self.Vessel_length/speed)**2 + 1)
		return factor*eddy0knots

	def ForwardSpeedPotential(self, PotentialDamp0kt, speed):
		self.PotentialDamp0kt = PotentialDamp0kt
		omega = self.wave_freq*(speed/self.g)
		zeta = self.wave_freq*self.wave_freq*self.Vessel_draft/self.g
		A1 = 1 + (zeta**(-1.2))*np.exp(-2*zeta)
		A2 = 0.5 + (zeta**(-1))*np.exp(-2*zeta)
		factor = 0.5*(A2 + 1 + (A2 - 1)*np.tanh(20*(omega-0.3)) + (2*A1 - A2 - 1)*np.exp(-150*(omega-0.25)**2)) - 1
		return factor*PotentialDamp0kt

	def LiftDamping(self, speed):
		if self.Cm > 0.97:
			kappa = 0.3
		elif self.Cm > 0.92:
			kappa = 0.1
		else:
			kappa = 0
		kn = 2*np.pi*self.Vessel_draft/self.Vessel_length + kappa*(4.1*self.Vessel_width/self.Vessel_length - 0.045)
		LiftDamping = (0.15/2)*self.water_density*speed*self.Vessel_length*(self.Vessel_draft**3)*kn*(1 - 2.8*self.OG/self.Vessel_draft + 4.667*(self.OG/self.Vessel_draft)**2)
		return LiftDamping

	def ForwardSpeedB44(self, speed, Pct_crit0kt, PotentialDamp0kt):
		self.PotentialDamp0kt = PotentialDamp0kt
		self.pct_crit = Pct_crit0kt
		speed = speed*0.514444444 # convert speed from knots to m/s
		self.B44e = self.ForwardSpeedEddy(Pct_crit0kt, speed)
		self.B44p = self.ForwardSpeedPotential(PotentialDamp0kt, speed)
		self.B44L = self.LiftDamping(speed)

		return self.B44e + self.B44p + self.B44L

	def PlotDampingGraph(self, speeds):
		B44e = [self.B44(self.pct_crit)]
		B44p = [0]
		B44L = [0]
		B44  = [0]

		for i in speeds:
			if i == 0: continue
			B44e.append(self.ForwardSpeedEddy(self.pct_crit, i*0.514))
			B44p.append(self.ForwardSpeedPotential(self.PotentialDamp0kt, i*0.514))
			B44L.append(self.LiftDamping(i*0.514))
			B44.append(B44e[-1] + B44p[-1] + B44L[-1])

		plt.style.use('seaborn-ticks') # Plot style
		plt.figure(1)
		plt.plot(speeds, 100*np.array(B44e)/self.crit_damping, label="Eddy damping only")
		plt.plot(speeds, 100*(np.array(B44e) + np.array(B44p))/self.crit_damping, label="+Potential damping")
		plt.plot(speeds, 100*(np.array(B44e) + np.array(B44p) + np.array(B44L))/self.crit_damping, label="+Lift damping")
		plt.legend()
		plt.ylabel("B44 damping [% crit]")
		plt.xlabel("Vessel speed [kt]")
		plt.savefig("Roll_damping_curve.png")
		# plt.figure(2)
		# plt.plot(speeds, B44e, label="Eddy damping")
		# plt.plot(speeds, np.array(B44p), label="Potential damping")
		# plt.plot(speeds, np.array(B44L), label="Lift damping")
		# plt.legend()
		# plt.ylabel("B44 damping [t*m^2]")
		# plt.xlabel("Vessel speed [kt]")

if __name__ == '__main__':
	L 	= 160.9 # [m] length of vessel	
	B 	= 49 # [m]
	D	= 5.198 # [m] draft of vessel
	VCG = 18.21 # [m] Vertical centre of gravity
	GMX	= 19.68 # [m] Transverse GM
	Troll = 13.9 # [s] Vessel roll period
	disp = 32314.9 # [t] Vessel displacement
	speed = 5.14 # [m/s] Vessel speed
	PotentialDamp0kt = 389830 # [t*m^2/s] Potential damping at 0kt @ Troll
	Cm = 0.99
	Pct_crit = 4.38

	# L 	= 160.9 # [m] length of vessel	
	# B 	= 49 # [m]
	# D	= 5.632 # [m] draft of vessel
	# VCG = 20.31 # [m] Vertical centre of gravity
	# GMX	= 15.27 # [m] Transverse GM
	# Troll = 15.6 # [s] Vessel roll period
	# disp = 35152.50 # [t] Vessel displacement
	# speed = 5.14 # [m/s] Vessel speed
	# PotentialDamp0kt = 184040 # [t*m^2/s] Potential damping at 0kt @ Troll
	# Cm = 0.99
	# Pct_crit = 7.39

	damping = IkedaAdditionalDamping(L, B, D, VCG, GMX, Troll, disp, Cm)
	print(damping)
	roll1 = damping.Roll_amplitude(6.06)
	roll2 = damping.Roll_amplitude(Pct_crit)
	rollfwd = damping.ForwardSpeedB44(speed, Pct_crit, PotentialDamp0kt)
	print("\n\nSignificant roll amplitude (@6.06% crit): {0:.2f} deg".format(roll1))
	print("Significant roll amplitude (@{0:.2f}% crit)".format(Pct_crit, damping.B44(Pct_crit)))	
	print("B44 eddy damping (@10kt): {0:.2f} t*m^2/s".format(damping.ForwardSpeedEddy(Pct_crit, speed)))	
	print("B44 potential damping (@10kt): {0:.2f} t*m^2/s".format(damping.ForwardSpeedPotential(PotentialDamp0kt, speed)))	
	print("B44 Lift damping (@10kt): {0:.2f} t*m^2/s".format(damping.LiftDamping(speed)))	
	print("Forward speed b44 damping (@10kt): {0:.2f} t*m^2/s".format(rollfwd))
	speeds = np.arange(1,20.1,1)
	damping.PlotDampingGraph(speeds)


	


