import numpy as np

a = np.random.uniform(-7.9103311e-9, 7.9103311e-7, size=(10,))
oom = -2
print(a)
print(np.round(np.log10(np.abs(a))))
print(oom - np.round(np.log10(np.abs(a))))
print(a * 10**(oom - np.round(np.log10(np.abs(a)))))
print(np.round(np.log10(np.abs(a * 10**(oom - np.round(np.log10(np.abs(a))))))))
