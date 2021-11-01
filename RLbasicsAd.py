class BernoulliBandit(object):
    def __init__(self, p):
        self.p = p
    def display_ad(self):
        reward = np.random.binomial(n=1, p=self.p)
        return reward

adA = BernoulliBandit(0.004)
adB = BernoulliBandit(0.016)
adC = BernoulliBandit(0.02)
adD = BernoulliBandit(0.028)
adE = BernoulliBandit(0.031)
ads = [adA, adB, adC, adD, adE]

