#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BE523 Biosystems Analysis & Design
HW16 - Questions 3. SIS model with born and death rates 

Created on Sat Feb 27 03:03:25 2021
@author: eduardo
"""
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, Trace
from sympy.matrices import Matrix

dt = 0.01  # time step, hours

alpha = 0.005  #rate of infection 1/person-hr
gamma = 0.02 # hrs for recovery
epsilon = 0.05 # initial number of infectives
steps = 1000  # time of simulation

N = 18  # total population
S0 = 10
I0 = N - S0

t = np.linspace(0, steps, int(steps/dt)+1)

I = np.zeros(len(t))
S = np.zeros(len(t))
I[0], S[0] = I0, S0

# Numerical solution, using Euler method
for i in range(1, len(t)):
    I[i] = I[i-1] + I[i-1] * (alpha * S[i-1] - gamma) * dt
    S[i] = S[i-1] + S[i-1] * (-alpha * I[i-1] + epsilon) * dt

# Use symbolic functions to compute derivatives and Jacobian
x = Symbol('x')  # Infectives
y = Symbol('y')  # Susceptibles
variables = Matrix([x, y])
# Create string representations of the functions f and g
f = '-' + str(alpha) + '*' + str('x') + '*' + str(y) + '+' + str(epsilon) + '*' + str(y)
g = str(alpha) + '*' + str('x') + '*' + str(y) + '-' + str(gamma) + '*' + str(x) 
functions = Matrix([[f, g]])

J = functions.jacobian(variables)  # Calculate Jacobian matrix

# Evaluate equilibrium points
I_equil = alpha/epsilon
S_equil = gamma/alpha
J0 = J.subs([(x, I_equil), (y, S_equil)])
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
plt.plot(t, S, label='S')
plt.plot(t, I, label='I')
plt.legend(loc='best')
plt.xlabel('Time (hours)')
plt.ylabel('Population')
plt.savefig('q3_SIS_model.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(1)
plt.plot(S, I)
plt.xlabel('S')
plt.ylabel('I')
plt.savefig('q3_SIS_model_phase.png', dpi=300, bbox_inches='tight')
plt.show()