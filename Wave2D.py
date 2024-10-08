import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

x, y, t = sp.symbols('x,y,t')

class Wave2D:
    """
    Dirichlet Wave
    """

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        self.h = 1.0 / N
        x = np.linspace(0, 1, self.N+1)
        self.xij, self.yij = np.meshgrid(x, x, indexing='ij', sparse=sparse)

    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D /= self.h**2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return sp.sqrt(self.c**2*((self.mx * sp.pi)**2 + (self.my * sp.pi)**2))

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, mx, my):
        """Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.Unp1, self.Un, self.Unm1 = np.zeros((3, self.N+1, self.N+1))
        self.mx = mx
        self.my = my
        
        self.Unm1[:] = sp.lambdify((x, y), self.ue(mx, my).subs({t: 0}))(self.xij, self.yij)
        self.Un[:] = sp.lambdify((x, y), self.ue(mx, my).subs({t: self.dt}))(self.xij, self.yij)

        #self.Un = self.Unm1 + 0.5*(self.c*self.dt)**2*(D @ self.Unm1 + self.Unm1 @ D.T)

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl*self.h / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        ue = sp.lambdify((x, y), self.ue(self.mx, self.my).subs({t: t0}))(self.xij, self.yij)
        error = u - ue
        return np.sqrt(self.h**2*np.sum(error**2))

    def apply_bcs(self):
        # Put to zero, since ue is zero on the boundaries
        self.Unp1[0, :] = 0
        self.Unp1[-1, :] = 0
        self.Unp1[:, -1] = 0
        self.Unp1[:, 0] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.create_mesh(N)
        self.cfl = cfl
        self.c = c
        
        # set u0 and u1
        D = self.D2()
        self.initialize(mx, my)

        # iterate over time steps
        plotdata = {0: self.Unm1.copy()}
        if store_data == 1:
            plotdata[1] = self.Un.copy()
        for n in range(2, Nt + 1):
            self.Unp1[:] = 2*self.Un - self.Unm1 + (c*self.dt)**2*(D @ self.Un + self.Un @ D.T)
            self.apply_bcs()

            if n % store_data == 0:
                plotdata[n] = self.Unp1.copy()

            # update solutions
            self.Unm1[:] = self.Un
            self.Un[:] = self.Unp1
        
        if store_data > 0:
            return plotdata
        if store_data == -1:
            return self.h, self.l2_error(self.Un, self.dt*Nt)
        else:
            return 1       

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err)
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D[0, :4] = -2, 2, 0, 0
        D[-1, -4:] = 0, 0, 2, -2
        D /= self.h**2
        return D

    def ue(self, mx, my):
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        pass

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(m=5,mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    mx = 3
    C = 1/np.sqrt(2)

    # Dirichlet
    sol = Wave2D()
    r, E, h = sol.convergence_rates(cfl=C, mx=mx, my=mx)
    assert np.all(E < 1e-12)

    # Neumann
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(cfl=C, mx=mx, my=mx)
    assert np.all(E < 1e-12)


test_convergence_wave2d()
test_convergence_wave2d_neumann()
test_exact_wave2d()

# Animation
wave = Wave2D_Neumann()
data = wave(N=40, Nt=301,mx=2, my=2, cfl=1/np.sqrt(2), store_data=1)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
frames = []
for n, val in data.items():
    frame = ax.plot_surface(wave.xij, wave.yij, val,  cmap=cm.coolwarm, linewidth=0)
    frames.append([frame])
ax.set_title(r'Neumann wave')
ax.text2D(0.05, 0.97, r'$u(t, x, y) = \cos(2 \pi x)\cos(2 \pi y)\cos(\omega t)$,      $C = \frac{1}{\sqrt{2}}$', transform=ax.transAxes)
ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True, repeat_delay=1000)
ani.save('report/neumannwave.gif', writer='pillow', fps=10)



