import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def E(x, alpha):
    return  np.sinc(alpha*x/np.pi)**2

def C(x, delta):
    return np.cos(delta*x)**2

def mu(x, B, I, alpha, delta):
    return B + I*E(x, alpha)*C(x, delta)

data = np.genfromtxt(
"data.csv",
delimiter=",",
names=True
)
x = data["x"]
y = data["y"]
sigma_y = data["sigma_y"]

def size_comparison (data):
    return (len(data["x"])==len(data["y"]) and len(data["y"])==len(data["sigma_y"]))



print(size_comparison(data))
print("Numero di conteggi: ", np.floor(np.sum(y)))
print("Intervallo di x : ", abs(x[0]-x[-1]))


B0 = (y[0] + y[-1])/2
I0 = np.max(y) - B0
delta0 = np.pi/0.6
alpha0 = np.pi/3.5

lower = [0.0, 0.0, 0.10, 1]
upper = [np.inf, np.inf, 3.0, 12.0]
p0 = [B0, I0, alpha0, delta0]

print("Parametri: ", p0)
print("Lower bounds: ", lower)
print("Upper bounds: ", upper)

popt, pcov = curve_fit(
mu, x, y,
p0=p0,
sigma=sigma_y,
absolute_sigma=True,
bounds=(lower, upper),
maxfev=1000000
)

B_hat = popt[0]
I_hat = popt[1]
alpha_hat = popt[2]
delta_hat = popt[3]

sigma_y_hat = np.sqrt(np.diagonal(pcov))

den = np.sqrt(np.outer(np.diagonal(pcov),np.diagonal(pcov)))
corr = pcov / den
print("Parametro B stimato: ",f"{B_hat:.1e}", " +/- ",f"{sigma_y_hat[0]:.1e}")
print("Parametro I stimato: ",f"{I_hat:.1e}", " +/- ",f"{sigma_y_hat[1]:.1e}")
print("Parametro alpha stimato: ",f"{alpha_hat:.1e}", " +/- ",f"{sigma_y_hat[2]:.1e}")
print("Parametro delta stimato: ",f"{delta_hat:.1e}", " +/- ",f"{sigma_y_hat[3]:.1e}")




fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
axes[0].scatter(x, y, color="tab:red", label="data")
axes[0].errorbar(x, y, sigma_y, fmt='o', color="tab:red")
axes[0].grid(True)
axes[0].legend()
axes[0].plot(x, mu(x, B_hat, I_hat, alpha_hat, delta_hat), color="tab:green", label="fit")

plt.imshow(corr, vmin=-1, vmax=1, cmap="plasma")
plt.colorbar(label="correlazione")
plt.xticks([0, 1, 2, 3], ["B", "I", "alpha", "delta"])
plt.yticks([0, 1, 2, 3], ["B", "I", "alpha", "delta"])
plt.title("Matrice di correlazione dei parametri")






#Da controllare la vicinanza ai bound




plt.tight_layout()
plt.show()