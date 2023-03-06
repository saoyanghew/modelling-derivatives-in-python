class Option:

    def __init__(self, Exercise, Type, price, strike, vol, rate, div, T):
        self.Exercise = Exercise
        self.Type = Type
        self.price = price
        self.strike = strike
        self.vol = vol
        self.rate = rate
        self.div = div
        self.T = T

    class OptionGreeks:

        def __init__(self):
            self.d1 = (np.log(self.price/self.strike) + (self.rate - self.div + self.vol * self.vol / 2) * T) / (self.vol * np.sqrt(self.T))
            
            self.d2 = self.d1 - self.vol * np.sqrt(self.T)

            self.normal_prime = np.random.normal(self.d1)

        def calcVega(self):
            """
            calcVega : calculates vega (sensitivity to volatility)
            [in] : price  : stock price
                   strike : strike price
                   rate   : interest rate
                   div    : dividend yield
                   vol    : volatility
                   T      : time to maturity
            [out]: vega
            """

            vega = (self.normal_prime * np.exp(- self.div * self.T)) * self.price * np.sqrt(self.T)

            return vega
        
        def calcDelta(self):
            """
            calcDelta : calculates delta (sensitivity to underlying stock price)
            [in] : price  : stock price
                   strike : strike price
                   rate   : interest rate
                   div    : dividend yield
                   vol    : volatility
                   T      : time to maturity
                   Type   : 'C'all or 'P'ut
            [out]: delta
            """
            if self.Type == 'C':
                delta = np.exp(- self.div * self.T) * self.normal_prime
            else:
                delta = np.exp(- self.div * self.T) * (self.normal_prime - 1)
            
            return delta

        def calcGamma(self):
            """
            calcGamma : calculates gamma (sensitivity to delta)
            [in] : price  : stock price
                   strike : strike price
                   rate   : interest rate
                   div    : dividend yield
                   vol    : volatility
                   T      : time to maturity
            [out]: gamma
            """
            gamma = (self.normal_prime * np.exp(- self.div * self.T)) / (self.price * self.vol * np.sqrt(self.T))

            return gamma

        def calcRho(self):
            """
            calcRho : calculates rho (sensitivity to risk-free rate)
            [in] : price  : stock price
                   strike : strike price
                   rate   : interest rate
                   div    : dividend yield
                   vol    : volatility
                   T      : time to maturity
                   Type   : 'C'all or 'P'ut
            [out]: rho
            """
            if self.Type == 'C':
                rho = self.strike * self.T * np.exp(- self.rate * self.T) * np.random.normal(self.d2)
            else:
                rho = - self.strike * self.T * np.exp(- self.rate * self.T) * np.random.normal(- self.d2)

            return rho

        def calcTheta(self):
            """
            calcTheta : calculates theta (sensitivity to time to maturity)
            [in] : price  : stock price
                   strike : strike price
                   rate   : interest rate
                   div    : dividend yield
                   vol    : volatility
                   T      : time to maturity
                   Type   : 'C'all or 'P'ut
            [out]: theta
            """
            if self.Type == 'C':
                theta = (- self.price * self.normal_prime * self.vol * np.exp(- self.div * self.T)) / (2 * np.sqrt(self.T)
                + self.div * self.price * self.normal_prime * np.exp(- self.div * self.T)
                - self.rate * self.strike * np.exp(- self.rate * self.T) * np.random.normal(self.d2)
            else:
                theta = (- self.price * self.normal_prime * self.vol * np.exp(- self.div * self.T)) / (2 * np.sqrt(self.T))
                - self.div * self.price * np.random.normal(-self.d1) * np.exp(- self.div * self.T)  
                + self.rate * self.strike * np.exp(- self.rate * self.T) * np.random.normal(- self.d2)

            return theta

class VanillaOption(Option):

    def __init__(self, Exercise, Type, price, strike, vol, rate, div, T):
        super().__init__(Exercise, Type, price, strike, vol, rate, div, T)


class BlackScholesOption(VanillaOption):
    def __init__(self, Exercise, Type, price, strike, vol, rate, div, T):
        super().__init__(Exercise, Type, price, strike, vol, rate, div, T)

    def setVolatility(self, vol_new):
        self.vol = vol_new
    
    def setRiskFreeRate(self rate_new):
        self.rate = rate_new

    def setDividendYield(self, div_new):
        self.div = div_new
    
    def calcBSCallPrice(self):
        """
        calcBSCallPrice : calculates Black-Scholes call price
            [in] : price  : stock price
                   strike : strike price
                   rate   : interest rate
                   div    : dividend yield
                   vol    : volatility
                   T      : time to maturity
            [out]: call price
        """
        d1 = (np.log(self.price/self.strike) + (self.rate - self.div + self.vol * self.vol / 2) * T) / (self.vol * np.sqrt(self.T))
        d2 = d1 - self.vol * np.sqrt(self.T)

        prob1 = np.random.normal(d1)
        prob2 = np.random.normal(d2)

        call_price = self.price * np.exp(- self.div * self.T) * prob1 - self.strike * np.exp(- self.rate * self.T) * prob2

        return call_price

    def calcBSPutPrice(self):
        """
        calcBSCallPrice : calculates Black-Scholes put price
            [in] : price  : stock price
                   strike : strike price
                   rate   : interest rate
                   div    : dividend yield
                   vol    : volatility
                   T      : time to maturity
            [out]: put price
        """

        d1 = (np.log(self.price/self.strike) + (self.rate - self.div + self.vol * self.vol / 2) * T) / (self.vol * np.sqrt(self.T))
        d2 = d1 - self.vol * np.sqrt(self.T)

        prob1 = np.random.normal(-d1)
        prob2 = np.random.normal(-d2)

        put_price = self.strike * np.exp(- self.rate * self.T) * prob2 - self.price * np.exp(- self.div * self.T) * prob1

        return put_price
