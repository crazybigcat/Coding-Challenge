import numpy as np
import c60

beta = 1
result = c60.get_lnz(beta=beta)
print('answer one = ' + str(result))
if beta < 0:
    print(np.log(2)/60 - beta * 90/60)
elif beta == 0:
    print(np.log(2))
beta1 = 9
beta2 = 10
result1 = c60.get_lnz(beta=beta1)
result2 = c60.get_lnz(beta=beta2)
E = 60 * (result2 - result1) / (beta1 - beta2)
print('ground energy = ' + str(E))
lnx = result1 * 60 + beta1 * E
x = np.exp(lnx)
print('简并度 = ' + str(x))
