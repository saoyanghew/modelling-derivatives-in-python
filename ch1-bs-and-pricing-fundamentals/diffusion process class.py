import numpy as np

class DiffusionProcess:
    """
    General diffusion process class
    This class describes a stochastic process governed by dx_t = mu(t, x_t) dt + sigma(t, x_t) dz_t
    """

    def __init__(self, x0_):
        self.x0 = x0_
    
    def drift(self, t, x):
        """
        Returns the drift part of the equation - i.e. mu(t, x_t)
        """
        return 0

    def diffusion(self, t, x):
        """
        Returns the diffusion part of the equation - i.e. sigma(t, x_t)
        """
        return 0
    
    def expectation(self, t0, dt):
        """
        Returns the expectation of a process after a time interval.
        i.e. E(x_{t_0 + delta t} | x_{t_0} = x_0), by default, this is an Euler approximation
        x_0 + mu(t_0, x_0) delta t
        """
        return self.x0 + self.drift(t0, self.x0) * dt
    
    def variance(self, t0, dt):
        """
        Returns the variance of a process after a time interval.
        i.e. Var(x_{t_0 + delta t} | x_{t_0} = x_0), by default, this is an Euler approximation
        sigma(t_0, x_0)^2 \Delta t
        """
        sigma = self.diffusion(t0, self.x0)
        return sigma * sigma * dt

class BlackScholesProcess(DiffusionProcess):
    """
    This class describes the stochastic process governed by dS = (r - 0.5 sigma^2) dt + sigma dzt
    """

    def __init__(self, x0_, r_, sigma_):
        super().__init__(x0_)
        self.r = r_
        self.sigma = sigma_
    
    def drift(self, t, x):
        return self.r - 0.5 * self.sigma * self.sigma
    
    def diffusion(self, t, x):
        return self.sigma

class OrnsteinUhlenbeckProcess(DiffusionProcess):
    """
    This class describes the stochastic process governed by dx = - a x_t dt + sigma dz_t
    """

    def __init__(self, x0_, speed_, volatility_):
        super().__init__(x0_)
        self.speed = speed_
        self.volatility = volatility_
    
    def drift(self, t, x):
        return - self.speed * x

    def diffusion(self, t, x):
        return self.volatility

    def expectation(self, t0, dt):
        return self.x0 * np.exp(-self.speed * dt)

    def variance(self, t0, dt):
        return 0.5 * self.volatility * self.volatility / self.speed * (1.0 - np.exp(-2.0 * self.speed * dt))
    
class SquareRootProcess(DiffusionProcess):
    """
    This class describes the stochastic process governed by dx = a (b - x_t) dt + sigma sqrt(x_t) dz_t
    """

    def __init__(self, x0_, mean_, speed_, volatility_):
        super().__init__(x0_)
        self.mean_ = mean_
        self.speed = speed_
        self.volatility = volatility_

    def drift(self, t, x):
        return self.speed * (self.mean - x)

    def diffusion(self, t, x):
        return self.volatility * np.sqrt*(x)
