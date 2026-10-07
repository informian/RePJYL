import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

c1, c2, c3 = 1, 0.1, 0.1
Lr = c2
n = 4
alpha, beta, ytM = 0.3, 0.5, 1.0
ita_min, ita_max = 0.005, 1
cnt_global = 0

P_half = np.random.randn(n,n)
P = P_half @ P_half.T
U, _ = np.linalg.qr(np.random.randn(n, n))
pre_d = np.random.rand(n)
pre_d = pre_d * 10.0 / np.max(pre_d) 
Q = U @ np.diag(pre_d) @ U.T

vec3 = c3 * np.ones(n)

x_hat_star = np.zeros(n)
x_star = np.linalg.solve(c1 * P + c2 * Q, c2 *  Q @ vec3)

def func_hat(x):    return 0.5 * c1 * x.T @ P @ x
def r_func(x):    return 0.5 * c2 * (x - vec3).T @ Q @ (x - vec3)
def func(x):    return func_hat(x) + r_func(x)
def grad_func_hat(x):    return c1 * P @ x
def grad_r(x):    return c2 * Q @ (x - vec3)
def grad_func(x):    return grad_func_hat(x) + grad_r(x)

def BTLS(x_btls, imn_btls, imx_btls, vecvar_blts):
    global cnt_global
    ita_btls = imx_btls
    cnt_global += 1
    const_fx = func(x_btls)
    const_nlnv = np.linalg.norm(vecvar_blts)

    while func(x_btls - ita_btls * vecvar_blts) > const_fx - alpha * ita_btls * ( const_nlnv ** 2 ):
        cnt_global += 1
        if ita_btls > imn_btls:
            ita_btls *= beta
        else:
            ita_btls = 0
            break
    
    return ita_btls

def MBD(x_mbd, delta_mbd, gamma, e0_line):
    global cnt_global
    x_plus_mbd = x_mbd.copy()
    g_tilde = grad_func_hat(x_mbd) + delta_mbd
    e_line = e0_line
    nfe, inac, mode = [], [], []

    ita = BTLS(x_mbd.copy(), ita_min, ita_max, g_tilde)
    x_plus_mbd = x_mbd - ita * g_tilde
    e_line += Lr * np.linalg.norm(x_plus_mbd - x_mbd)

    nfe.append(cnt_global)
    inac.append(np.linalg.norm(x_plus_mbd - x_star))
    mode.append('model')

    while e_line <= (1 - gamma) / gamma * np.linalg.norm(g_tilde) and ita != 0:

        x_mbd = x_plus_mbd.copy()
        g_tilde = grad_func_hat(x_mbd) + delta_mbd
        ita = BTLS(x_mbd.copy(), ita_min, ita_max, g_tilde)
        x_plus_mbd = x_mbd - ita * g_tilde
        e_line += Lr * np.linalg.norm(x_plus_mbd - x_mbd)
        nfe.append(cnt_global)
        inac.append(np.linalg.norm(x_plus_mbd - x_star))
        mode.append('model')

    return x_plus_mbd, nfe, inac, mode
        
def GDC(gamma, iters = 1000):
    global cnt_global
    cnt_global = 0
    nfe, inac, mode = [], [], []
    x_gdc = x_hat_star.copy()
    e0_line = 0
    delta_gdc = grad_r(x_gdc)

    nfe.append(cnt_global)
    inac.append(np.linalg.norm(x_gdc - x_star))
    mode.append('model')
    
    for _ in range(iters):

        x_gdc, _nfe, _inac, _mode = MBD(x_gdc.copy(), delta_gdc, gamma, e0_line)
        nfe.extend(_nfe)
        inac.extend(_inac)
        mode.extend(_mode)

        cnt_global += n
        delta_gdc = grad_r(x_gdc)
        ita = BTLS(x_gdc.copy(), 0, ita_max, grad_func(x_gdc))
        x_plus_gdc = x_gdc - ita * grad_func(x_gdc)
        e0_line = Lr * np.linalg.norm(x_plus_gdc - x_gdc)
        x_gdc = x_plus_gdc.copy()

        nfe.append(cnt_global)
        inac.append(np.linalg.norm(x_gdc - x_star))
        mode.append('modelfree')

        if np.linalg.norm(x_gdc - x_star) < 1e-3:
            break

    return nfe, inac, mode

def GDC0(iters=1000):
    global cnt_global
    
    x_gcd0 = x_hat_star.copy()
    nfe, inac = [], []
    cnt_global = 0
    
    for _ in range(iters):
        if np.linalg.norm(x_gcd0 - x_star) < 1e-3:
            break

        cnt_global += n
        ita = BTLS(x_gcd0, 0, ita_max, grad_func(x_gcd0))#BTLS(x_btls, imn_btls, imx_btls,gamma):
        x_gcd0 = x_gcd0 - ita * grad_func(x_gcd0)
        
        nfe.append(cnt_global)
        inac.append(np.linalg.norm(x_gcd0 - x_star))
        
    return nfe, inac

if __name__ ==  "__main__":
    nfe_6, inac_6, mode_6 = GDC(gamma=0.6)
    nfe_9, inac_9, mode_9 = GDC(gamma=0.9)
    nfe_0, inac_0 = GDC0()
    fig, axs = plt.subplots(1, 3, figsize=(24, 5))

    axs[0].set_title("gamma=0.6")
    for i in range(len(nfe_6)-1):
        color = 'red' if mode_6[i] == 'model' else 'blue'
        axs[0].plot(nfe_6[i:i+2], inac_6[i:i+2], color=color)
    axs[0].set_yscale('log')
    axs[0].set_xlabel("Number of Function Evaluations")
    axs[0].set_ylabel("||x - x*||")

    axs[1].set_title("gamma=0.9")
    for i in range(len(nfe_9)-1):
        color = 'red' if mode_9[i] == 'model' else 'blue'
        axs[1].plot(nfe_9[i:i+2], inac_9[i:i+2], color=color)
    axs[1].set_yscale('log')
    axs[1].set_xlabel("Number of Function Evaluations")
    axs[1].set_ylabel("||x - x*||")

    axs[2].set_title("Model-Free Method")
    axs[2].plot(nfe_0, inac_0, color='blue')
    axs[2].set_yscale('log')
    axs[2].set_xlabel("Number of Function Evaluations")
    axs[2].set_ylabel("||x - x*||")

    plt.tight_layout()
    plt.show()