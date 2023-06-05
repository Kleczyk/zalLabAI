import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

Z = ctrl.Antecedent(np.arange(0,100.1,0.1), 'Z')
psi = ctrl.Antecedent(np.arange(-np.pi ,np.pi+0.01,0.01), 'psi')
VL = ctrl.Consequent(np.arange(-20000,20001,1), 'VL')
VR = ctrl.Consequent(np.arange(-20000,20001,1), 'VR')

Z['NR'] = fuzz.trimf(Z.universe, [0,0,100])
Z['FR'] = fuzz.trimf(Z.universe, [0,100,100])


psi['N'] = fuzz.trimf(psi.universe, [-np.pi, -np.pi ,0])
psi['Z'] = fuzz.trimf(psi.universe, [-np.pi,0,np.pi])
psi['P'] = fuzz.trimf(psi.universe, [0,np.pi,np.pi])

VR['B'] = fuzz.trimf(VR.universe, [-20000,-20000,0])
VR['S'] = fuzz.trimf(VR.universe, [-20000,0,20000])
VR['F'] = fuzz.trimf(VR.universe, [0,20000,20000])

VL['B'] = fuzz.trimf(VL.universe, [-20000,-20000,0])
VL['S'] = fuzz.trimf(VL.universe, [-20000,0,20000])
VL['F'] = fuzz.trimf(VL.universe, [0,20000,20000])

Z.view()
psi.view()
VL.view()
VR.view()

regula1 = ctrl.Rule(antecedent=(Z['NR'] & psi['N']), consequent=(VL['B'] , VR['F']))
regula2 = ctrl.Rule(antecedent=(Z['NR'] & psi['Z']), consequent=(VL['S'] , VR['S']))
regula3 = ctrl.Rule(antecedent=(Z['NR'] & psi['P']), consequent=(VL['F'] , VR['B']))
regula4 = ctrl.Rule(antecedent=(Z['FR'] & psi['N']), consequent=(VL['B'] , VR['F']))
regula5 = ctrl.Rule(antecedent=(Z['FR'] & psi['Z']), consequent=(VL['F'] , VR['F']))
regula6 = ctrl.Rule(antecedent=(Z['FR'] & psi['P']), consequent=(VL['F'] , VR['B']))



F_ctr = ctrl.ControlSystem([regula1,regula2,regula3,regula4,regula5,regula6])
F_sym = ctrl.ControlSystemSimulation(F_ctr)



F_sym.input['Z']= 60
F_sym.input['psi']= np.pi/3



F_sym.compute()
print("output: ", F_sym.output['VR'])
print("output: ", F_sym.output['VL'])
VR.view(sim=F_sym)


# n_points = 10
# upsampledX = np.linspace(0, 100, n_points)
# upsampledY = np.linspace(-np.pi, np.pi, n_points)

# x, y = np.meshgrid(upsampledX, upsampledY)
# z = np.zeros((len(x),len(y)))

# for i in range(n_points):
#     for j in range(n_points):
#         F_sym.input['Z'] = x[i, j]
#         F_sym.input['psi'] = y[i, j]
#         F_sym.compute()
#         z[i, j] = F_sym.output['VR']

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(x, y, z, cmap='viridis')
# ax.set_xlabel('z')
# ax.set_ylabel('psi')
# ax.set_zlabel('vr')
# ax.view_init(30, 200)
# print("Najmniejsza możliwa otrzymana wartość CC: ",z.min())
# print("Największa możliwa otrzymana wartość CC: ",z.max())
# plt.show()
pass