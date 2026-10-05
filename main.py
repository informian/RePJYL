import numpy as np
import matplotlib.pyplot as plt

#initial number:(data from 5.1)
c1 = 1
c2 = c3 = 0.1
Lr = c2
alpha, beta, ytM, ytm, gamma = 0.3, 0.5, 1.0, 0.005, 0.5 #yt means yita!M for max and m for min
n = 4

#initial matrics
P_12 = np.random.randn(n,n)
P = P_12 @ P_12.T #make a SPD
U, _ = np.linalg.qr(np.random.randn(n, n))#division
pre_lamda = np.random.rand(n)
pre_lamda = pre_lamda * 10 / np.max(pre_lamda)
lamda = np.diag(pre_lamda)
Q = U @ lamda @ U.T

x_actr = np.linalg.solve(c1 * P + c2 * Q, c2 *  Q @ (c3 * np.ones(n)))
np.random.seed(40)

#functions defined
def f_hat(x):
    return c1 * x.T @ P @ x / 2
def r_func(x):
    return c2 * (x - c3).T @ Q @ (x - c3) /2
def func(x):
    return float( f_hat(x) + r_func(x) )

#gradient function:(I can not compute grad of a matrix so the result is from deepseek)
def grad_f_hat(x):
    return c1 * (P + P.T) @ x / 2
def grad_r_func(x):
    return c2 * Q @ (x - c3)
def grad_func(x):
    return grad_f_hat(x) + grad_r_func(x)

def btls(x, tri_delta, bt_ytm):
    ytr = ytM #r for require(d)
    cnt = 1
    const_func_numb = func(x)
    while func(x - ytr * tri_delta) > const_func_numb - alpha * ytr * np.linalg.norm(tri_delta)**2:
        cnt += 1
        if ytr > bt_ytm :
            ytr *= beta
        else:
            ytr = 0
            break
    return ytr, cnt

def ModelBasedDescent(x, delta, gamma, line_e0):
    # init:
    self_ytm = 0.005
    cnt = 0
    x_plus = x.copy()
    g_tilde = grad_f_hat(x) + delta
    line_e = line_e0
    local_nfe, local_inac, local_mode = [], [], []

    while line_e <= (1 - gamma) / gamma * np.linalg.norm(g_tilde):
        x = x_plus.copy()
        g_tilde = grad_f_hat(x) + delta
        eta, cnt_s = btls(x, g_tilde, self_ytm)
        cnt += cnt_s
        if eta == 0:
            break
        x_plus = x - eta * g_tilde
        line_e = Lr * np.linalg.norm(x_plus - x) + line_e 
        local_nfe.append(cnt)
        local_inac.append(np.linalg.norm(x_plus - x_actr))
        local_mode.append('model')

    return x, local_nfe, local_inac, local_mode

def GDC(gamma, iters=1000):
    x = np.zeros(n)
    delta = grad_r_func(x)
    line_e0 = 0.0
    cnt_box = [0]
    ytm = 0.005
    history_nfe, history_inac, history_mode = [], [], []
    history_nfe.append(cnt_box[0])
    history_inac.append(np.linalg.norm(x - x_actr))
    history_mode.append('model')

    for i in range(iters):
        if np.linalg.norm(x - x_actr) < 1e-3:
            break
        x, local_nfe, local_inac, local_mode = ModelBasedDescent(x, delta, gamma, line_e0)
        history_nfe.extend(local_nfe)
        history_inac.extend(local_inac)
        history_mode.extend(local_mode)
        if np.linalg.norm(x - x_actr) < 1e-3:
            break
        cnt_box[0] += n
        grad_exact = grad_func(x)
        eta, cnt_s = btls(x, grad_exact, 0.0)
        cnt_box[0] += cnt_s
        x_old = x.copy()
        x = x - eta * grad_exact
        history_nfe.append(cnt_box[0])
        history_inac.append(np.linalg.norm(x - x_actr))
        history_mode.append('modelfree')
        delta = grad_exact - grad_f_hat(x)
        line_e0 = Lr * np.linalg.norm(x - x_old)

    return history_nfe, history_inac, history_mode

    #int main():
if __name__ == "__main__":
    nfe_06, inac_06, mode_06 = GDC(gamma=0.6)
    nfe_09, inac_09, mode_09 = GDC(gamma=0.9)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].set_title("GC Algorithm (gamma=0.6)")
    for i in range(len(nfe_06)-1):
        color = 'red' if mode_06[i] == 'model' else 'blue'
        axs[0].plot(nfe_06[i:i+2], inac_06[i:i+2], color=color)
    axs[0].set_yscale('log')
    axs[0].set_xlabel("Number of Function Evaluations")
    axs[0].set_ylabel("||x - x*||")

    axs[1].set_title("GC Algorithm (gamma=0.9)")
    for i in range(len(nfe_09)-1):
        color = 'red' if mode_09[i] == 'model' else 'blue'
        axs[1].plot(nfe_09[i:i+2], inac_09[i:i+2], color=color)
    axs[1].set_yscale('log')
    axs[1].set_xlabel("Number of Function Evaluations")
    axs[1].set_ylabel("||x - x*||")

    plt.show()