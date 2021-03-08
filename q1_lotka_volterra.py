#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BE523 Biosystems Analysis & Design
HW16 - Question 1. Lotka-Volterra model for owls and voles
       Find if equilibrium points are stable or not.

Created on Fri Mar  5 00:15:20 2021
@author: eduardo
"""
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Trace
from sympy.matrices import Matrix

dt = 1
steps = 500

delta = 0.02
alpha = 0.06
gamma = 1e-3
epsilon = 2e-4
V0, O0 = 40, 40  # initial population of voles and owls

t = np.linspace(0, steps, int(steps/dt)+1)
V = np.zeros(len(t))
O = np.zeros(len(t))
V[0], O[0] = V0, O0

for i in range(1, len(t)):
    V[i] = V[i-1] + V[i-1] * (alpha - gamma * O[i-1]) * dt
    O[i] = O[i-1] + O[i-1] * (epsilon * V[i-1] - delta) * dt

# Use symbolic functions to compute derivatives and Jacobian
v, o = symbols('v o')  # Voles and owls symbols
variables = Matrix([v, o])
# Create string representations of the functions f and g
# f = '0.06*v - 1e-3*v*o'
# g = '0.02*o + 2e-4*v*o'
# This updates the string equations, in case the parameters change
f = str(alpha) + '*' + str(v) + '-' + str(gamma) + '*' + str(v) + '*' + str(o)
g = str(delta) + '*' + str(o) + '+' + str(epsilon) + '*' + str(v) + '*' + str(o)
print("Equations: f(v,o)=", f, ' and g(v,o)=', g)
functions = Matrix([[f, g]])

J = functions.jacobian(variables)  # Calculate Jacobian matrix

# Evaluate equilibrium points
V_equil = delta/epsilon
O_equil = alpha/gamma
J0 = J.subs([(v, V_equil), (o, O_equil)])
# Calculate determinant and trace of the evaluated Jacobian matrix
det = J0.det()
trace = Trace(J0).simplify()
print('Determinant:', det, 'Trace:', trace)

if det > 0 and trace < 0:
    print('Point is stable')
elif det > 0 and trace > 0:
    print('Point is a repeller')
elif det > 0 and trace == 0:
    print('Point is unstable')
else:
    print('Unknown')

plt.figure(0)
plt.plot(t, V, 'b-', label='V')
plt.plot(t, O, 'r-', label='O')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend(loc='best')
# plt.savefig('q1_lotka_volterra_%dsteps.png' % steps, dpi=300, bbox_inches='tight')
plt.show()

plt.figure(1)
plt.plot(V, O)
plt.xlabel('V')
plt.ylabel('O')
# plt.savefig('q1_lotka_volterra_phase_%dsteps.png' % steps, dpi=300, bbox_inches='tight')
plt.show()
