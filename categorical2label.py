import numpy as np
def categorical2label(Y,textlabel=False):
    if not(textlabel):
        x = [np.argmax(Y[x]) for x in range(0, len(Y))]
    else:
        x = [str(np.argmax(Y[x])) for x in range(0, len(Y))]

    return x
