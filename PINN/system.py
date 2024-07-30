import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class MassSpringDamper:

    def __init__(self, m:float, k:float, c:float, F_ext:float=0) -> None:
        self.m = m
        self.k = k
        self.c = c
        self.F_ext = F_ext

    def f (self, t, y):
        x, v = y
        a = (self.F_ext - self.k*x - self.c*v) / self.m
        dydt = [v, a]
        return dydt
    
    def solve_system(self,y0:list,t0:float, tf:float, dt:int):
        y0 = [1, 0]
        t_eval = np.linspace(t0, tf, int(tf/dt))
        self.sol = solve_ivp(self.f, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval)
    
    def get_solution(self) -> np.array:
        return self.sol
    
    def show_plot(self) -> None:
        t = self.sol.t
        x = self.sol.y[0]
        v = self.sol.y[1]
        plt.figure()
        plt.plot(t, x, label='Displacement')
        plt.plot(t, v, label='Velocity')
        plt.title('Mass Spring Damper system')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True)
        plt.show()

def main() -> None:

    massSpringDamper = MassSpringDamper(1,2,0.5)
    massSpringDamper.solve_system(y0=[0,1],t0=0,tf=30,dt=0.1)
    sol = massSpringDamper.get_solution()


if __name__ == '__main__':
    main()