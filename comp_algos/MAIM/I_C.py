import matplotlib.pyplot as plt

def I_C_relation (Inf, Col, marginalGain):
    l = len(Inf)
    plt.figure()
    plt.plot(range(l),Inf)
    plt.plot(range(l),Col)
    plt.bar(range(l),marginalGain)
    plt.show()
    pass
