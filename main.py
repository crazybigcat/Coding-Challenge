import numpy as np
import c60

beta = 1
result = c60.get_lnz(beta=beta)
print(result)
if beta <= 0:
    print(np.log(2)/60 - beta * 90/60)
