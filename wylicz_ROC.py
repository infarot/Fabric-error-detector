import pickle
import numpy as np
import matplotlib.pyplot as plt

# with open('wyniki.pkl', 'rb') as plik:
#     wyniki= pickle.load(plik)


def wylicz_ROC(wyniki, czy_rysowac=True):
    wyniki = np.asarray(wyniki)

    score_u = np.unique(wyniki[:,0])[::-1]
    n=len(score_u)

    total_0 = sum(wyniki[:,1]==0)
    total_1 = sum(wyniki[:,1]==1)

    AUC = 0
    KS =0
    tabela = np.zeros((n+1,2))
    for i in range(n):
        pomoc_0 = np.logical_and(wyniki[:, 0] >= score_u[i], wyniki[:, 1] == 0)
        pomoc_1 = np.logical_and(wyniki[:, 0] >= score_u[i], wyniki[:, 1] == 1)

        tabela[i+1,:2] = [sum(pomoc_0 )/total_0, sum(pomoc_1 )/total_1]
        AUC += 0.5*(tabela[i+1,1] + tabela[i,1]) * (tabela[i+1,0] - tabela[i,0])
        KS = max(KS, abs(tabela[i+1,1] - tabela[i+1,0]))


    np.savetxt('wyniki.txt',wyniki)

    if czy_rysowac:
        plt.plot(tabela[:,0],tabela[:,1])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(" AUC = {}".format(AUC))
        plt.show()
