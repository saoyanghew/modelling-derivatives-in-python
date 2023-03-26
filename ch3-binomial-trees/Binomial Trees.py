import numpy as np

def buildBinomialTreeAmerican(price, strike, rate, div, vol, T, N, type):
    """
    buildBinomialTreeCRRAmerican : computes the value of an American option using backwards induction in a CRR binomial tree
    [in]: 
        price : asset price
        strike : strike price
        rate : risk-free interest rate
        div : dividend yield
        vol : volatility
        T : time to maturity
        N : number of time steps
        exercise : European (E) or American (A)
        type : Call (C) or Put (P)
    [out]: value of American option
    """

    # initialisation
    S = np.zeros((200, 200)) # stock price matrix
    C = np.zeros((200, 200)) # call price matrix
    dt = T / N # time step size
    up = np.exp(vol * np.sqrt(dt)) # up movement
    down = 1 / up # down movement
    a = np.exp((rate - div) * dt) # growth rate in prob
    prob = (a - down) / (up - down)

    # compute stock prices at each node
    # initialise call prices
    for i in range(N):
        for j in range(i):
            S[i][j] = price * np.power(up, j) * np.power(down, i - j)
            C[i][j] = 0

    # compute terminal payoffs
    for i in reversed(range(N)):
        if type == 'C':
            C[N][j] = max(S[N][j] - strike, 0)
        else:
            C[N][j] = max(strike - S[N][j], 0)

    # work backwards
    for i in reversed(range(N-1)):
        for j in reversed(range(i)):
            C[i][j] = np.exp(- rate * dt) * prob * (C[i+1][j+1]) + (1 - prob) * C[i+1][j]
            
            if type == 'C':
                C[i][j] = max(S[i][j] - strike, C[i][j])
            else: 
                C[i][j] = max(strike - S[i][j], C[i][j])

    return C[0][0]

def buildTwoVarBinomialTree(S1, S2, strike, rate, div1, div2, rho, vol1, vol2, T, N, exercise, type):
    """
    buildTwoVarBinomialTree : computes the value of an American option using backwards induction in a CRR binomial tree
    [in]: 
        S1 : asset price 1
        S2 : asset price 2
        strike : strike price
        rate : risk-free interest rate
        div1 : dividend yield 1
        div2 : dividend yield 2 
        rho : correlation of asset 1 and asset 2
        vol1 : volatility 1
        vol2 : volatility 2
        T : time to maturity
        N : number of time steps
        exercise : European (E) or American (A)
        type : Call (C) or Put (P)
    [out]: value of American option
    """

    # initialise parameters
    dt = T / N
    mu1 = rate - div1 - 0.5 * vol1 ** 2
    mu2 = rate - div2 - 0.5 * vol2 ** 2
    dx1 = vol1 * np.sqrt(dt)
    dx2 = vol2 * np.sqrt(dt)

    # initialise stock and option price
    S1t = np.zeros((100))
    S2t = np.zeros((100))
    C = np.zeros((100, 100))

    # compute probabilities
    puu = (dx1 * dx2 + (dx2 * mu1 + dx1 * mu2 + rho * vol1 * vol2) * dt) / (4 * dx1 * dx2)
    pud = (dx1 * dx2 + (dx2 * mu1 - dx1 * mu2 - rho * vol1 * vol2) * dt) / (4 * dx1 * dx2)
    pdu = (dx1 * dx2 + (dx2 * mu1 + dx1 * mu2 - rho * vol1 * vol2) * dt) / (4 * dx1 * dx2)
    pdd = (dx1 * dx2 + (dx2 * mu1 - dx1 * mu2 + rho * vol1 * vol2) * dt) / (4 * dx1 * dx2)

    # initialise asset prices at maturity
    S1t[-N] = S1 * np.exp(- N * dx1)
    S2t[-N] = S2 * np.exp(- N * dx2)

    # compute stock prices at each node
    for j in range(-N+1, N):
        S1t[j] = S1t[j-1] * np.exp(dx1)
        S2t[j] = S2t[j-1] * np.exp(dx2)

    # compute early exercise payoff at each node
    for j in range(-N, N, 2):
        for k in range(-N, N, 2):
            if type == 'C':
                C[j][k] = max(0.0, S1t[j] - S2t[k] - strike)
            else:
                C[j][k] = max(0.0, strike - S1t[j] + S2t[k])

    # step back through the tree applying early exercise
    for i in reversed(range(N-1)):
        for j in range(-i, i, 2):
            for k in range(-i, i, 2):
                # compute risk neutral price
                C[j][k] = np.exp(- rate * T) * (pdd * C[j-1][k-1] + pud * C[j+1][k-1] + pdu * C[j-1][k+1] + puu * C[j+1][k+1])
                
                if exercise == 'A':
                    if type == 'C':
                        C[j][k] = max(C[j][k], S1t[j] - S2t[k] - strike)
                    else:
                        C[j][k] = max(C[j][k], strike - S1t[j] + S2t[k])
    
    return C[0][0]

def calcConvertibleBond(price, vol, rates, dividend, T, principal, couponRate, frequency, N, 
                        conversionRatio, conversionPrice, creditSpread, callSchedule):
    """
    calcConvertibleBond: computes the value of convertible bond with callable provisions
    [in]: 
        price : stock price
        vol : stock volatility
        rates : contains zero-curve rates
        dividend : stock dividend yield
        T : time to maturity of convertible bond
        principal : par value of bond
        couponRate : coupon rate of bond
        frequency : frequency of coupon payments
        N : number of time steps
        conversionRatio : conversion ratio
        conversionPrice : conversion price
        creditSpread : credit spread of issuer
        callSchedule : call schedule map of times to call prices
    [out] double
    """

    # initialisation
    up = 0.0
    down = 0.0
    interest = 0.0
    H = 0.0
    rate = rates[len(rates) - 1]
    dt = np.exp(vol * np.sqrt(dt))
    down = 1/up
    S = np.zeros((N, N))
    V = np.zeros((N, N))
    cp = np.zeros((N, N))
    call = np.zeros((N, N))

    # build CRR stock tree
    for i in range(N):
        for j in range(i):
            S[i][j] = price * np.power(up, j) * np.power(down, i-j)

    interest = principal * couponRate * dt

    for j in reversed(range(N)):
        payment = principal + principal * couponRate * dt
        if S[N][j] >= conversionPrice:
            V[N][j] = max(conversionRatio * S[N][j], payment)
        else:
            V[N][j] = payment

        if V[N][j] == conversionRatio * S[N][j]:
            cp[N][j] = 1.0
        else:
            cp[N][j] = 0.0

    # work backwards
    for i in reversed(range(N-1)):
        for j in reversed(range(i)):
            # compute price at current node
            call[i][j] = callSchedule[i]

            # compute conversion probability
            cp[i][j] = 0.5 * (cp[i+1][j+1] + cp[i+1][j])
            
            # compute credit adjusted discount rate
            creditAdjustedRate = cp[i][j] * rates[i] + (1 - cp[i][j]) * creditSpread

            # compute holding value
            H = 0.5 * ((V[i+1][j+1] + interest) / (1 + creditAdjustedRate * dt) + (V[i+1][j] + interest) / (1 + creditAdjustedRate * dt))

            # check if stock price exceeds conversion price
            if S[i][j] >= conversionPrice:
                V[i][j] = max(conversionRatio * S[i][j], min(H, call[i][j] + interest))
            else:
                V[i][j] = min(H, call[i][j] + interest)
    
    return V[0][0]