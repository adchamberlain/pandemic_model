# SIR (susceptible, infected, recovered) epidemic model.
# Based on https://www.nber.org/papers/w26867.pdf
# and https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/.
# March 2020

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 1000

# Initial number of exposed, infected and recovered individuals, I0 and R0.
E0, I0, R0 = 1, 1, 0

# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

# Contact rate (beta), mean recovery rate (gamma, in 1/days), and exposure to infected rate (sigma, 1/incubation period in days).
beta, gamma, sigma = 0.2, 1./18, 1./5.2

# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    
    dSdt = -beta * S * I / N
    
    dEdt = beta * S * I / N - sigma * E
    
    dIdt = sigma * E - gamma * I
    
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, E0, I0, R0

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, sigma))
S, E, I, R = ret.T

# Plot the data on four separate curves for S(t), E(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, E/1000, 'y', alpha=0.5, lw=2, label='Exposed')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()