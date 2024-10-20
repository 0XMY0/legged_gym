import numpy as np
from math import exp
import torch
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit

class LIPM:
    def __init__(self, height, freq, disy, phasediff, swingphasespan):
        self.height = height
        self.disy = disy
        self.g = 9.81
        self.w0 = (self.g/self.height)**0.5
        self.freq = freq # for a whole cycle containing two steps
        self.phasediff = phasediff # phase difference between two feet
        self.swingphasespan = swingphasespan # phase span for swing phase
        self.nseg = 200

    def _getswingsignal(self, phase):
        def _getphase(x, mu_A, mu_B):
            def normal_cdf(x, mu, sigma):
                return 0.5 * (1 + erf((x - mu) / (sigma * 2.0)))
            P1 = normal_cdf(x, mu_A, 0.01) * (1 - normal_cdf(x, mu_B, 0.01))
            if(mu_A > 0.2 and mu_B < 0.8):
                return P1
            if(mu_A <= 0.2):
                P2 = normal_cdf(x - 1, mu_A, 0.01) * (1 - normal_cdf(x - 1, mu_B, 0.01))
                return P1 + P2
            if(mu_B >= 0.8):
                P3 = normal_cdf(x + 1, mu_A, 0.01) * (1 - normal_cdf(x + 1, mu_B, 0.01))
                return P1 + P3
            return P1
        return _getphase(phase, 0, self.swingphasespan), _getphase(phase, self.phasediff, self.phasediff + self.swingphasespan)
    
    def _getzmp(self, phase):
        swingl, swingr = self._getswingsignal(phase)
        stancel = 1 - swingl
        stancer = 1 - swingr
        return self.disy * stancel + -self.disy * stancer
    
    def _getqend(self, q0):
        dt = 1.0 / self.freq / self.nseg
        q = q0
        qslog = []
        for i in range(self.nseg):
            zmp = self._getzmp(i / self.nseg)
            qslog.append([q, zmp])
            q = (q - zmp) * exp(self.w0 * dt) + zmp
        return q, qslog
    
    def _getq0y(self):
        q0 = 0.0
        f_grad_last = 10
        step = 1e-3
        while(1):
            qend, _ = self._getqend(q0)
            qend_add, _ = self._getqend(q0 + 1e-5)
            qend_sub, _ = self._getqend(q0 - 1e-5)
            qend_grad = (qend_add - qend_sub) / 2e-5
            f_grad = 2 * (qend - q0) * (qend_grad - 1)
            if(abs(f_grad) > abs(f_grad_last) or (f_grad * f_grad_last < 0)):
                step /= 2
            q0 -= np.clip(f_grad * step, -0.01, 0.01)
            f_grad_last = f_grad
            # print(q0, f_grad)
            if(abs(f_grad) < 1e-5):
                break
        return q0
    
    def _gettra(self):
        q0 = self._getq0y()
        qend, qslog = self._getqend(q0)
        Kp = 10.0
        y = [0, 0]
        y[1] = (q0 - y[0]) * self.w0
        ys = []
        for i in range(0, 3 * self.nseg):
            step, ii = divmod(i, self.nseg)
            zmpd = qslog[ii][1]
            qnow = y[0] + y[1] / self.w0
            qd = qslog[ii][0]
            zmp = -Kp * (qnow - qd) + zmpd
            dt = 1.0 / self.freq / self.nseg
            ddy = self.g / self.height * (zmp - y[0])
            y[0] += y[1] * dt + 0.5 * ddy * dt * dt
            y[1] += ddy * dt
            if step == 2:
                ys.append([i * dt, y[0].copy(), y[1].copy(), ddy, zmp, zmpd])
        return ys

def generatedata():
    freq_bounds = [0.5, 2.5]
    phasediff_range = [0.0, 0.5]
    swingphasespan_range = [0.3, 0.8]
    len = 1000
    data = []
    for i in range(len):
        lipm = LIPM(
            height=0.55,
            freq=np.random.uniform(freq_bounds[0], freq_bounds[1]),
            disy=0.1,
            phasediff=np.random.uniform(phasediff_range[0], phasediff_range[1]),
            swingphasespan=np.random.uniform(swingphasespan_range[0], swingphasespan_range[1]))
        print(i, 'freq = ', lipm.freq, 'phasediff = ', lipm.phasediff, 'swingphasespan = ', lipm.swingphasespan)
        vel = np.array(lipm._gettra())[:, 2]
        datanow = [vel[0], vel[int(lipm.nseg / 6)], vel[int(lipm.nseg / 3)], vel[int(lipm.nseg / 2)], vel[int(lipm.nseg / 3 * 2)], vel[int(lipm.nseg / 6 * 5)]]
        data.append([lipm.freq, lipm.phasediff, lipm.swingphasespan] + datanow)
        with open('data.csv', 'a') as f:
            f.write(','.join(map(str, data[-1])) + '\n')
    return data

def _fit_function2(x, a0, a1, a2, a3):
    return a0 + a1 * x[:, 0] + a2 * x[:, 1] + a3 * x[:, 2]

def fitdata2(x, y):
    param = curve_fit(_fit_function2, x, y)
    return param

# # test _getswingsignal and _getzmp
# lipm = LIPM(height=0.55, freq=1.5, disy=0.1, phasediff=0.5, swingphasespan=0.45)
# i = torch.linspace(0, 1, 1000)
# swingl, swingr = lipm._getswingsignal(i)
# # plot swingl, swingr in subplot 1
# fig, axs = plt.subplots(2)
# axs[0].plot(i, swingl, label='swingl')
# axs[0].plot(i, swingr, label='swingr')
# axs[0].legend()
# # plot zmp in subplot 2
# zmp = lipm._getzmp(i)
# axs[1].plot(i, zmp, label='zmp')
# axs[1].legend()
# plt.show()

# test _getqend and _getq0y
# lipm = LIPM(height=0.55, freq=1.5, disy=0.1, phasediff=0.5, swingphasespan=0.45)
# q0 = lipm._getq0y()
# qend, qslog = lipm._getqend(q0)
# # plot qslog in single plot
# fig, axs = plt.subplots(1)
# axs.plot(qslog, label='q')
# axs.legend()
# plt.show()

# test _gettra
# lipm = LIPM(height=0.55, freq=1.5, disy=0.1, phasediff=0.1, swingphasespan=0.45)
# ys = lipm._gettra()
# # plot ys in 2*2 subplots
# fig, axs = plt.subplots(2, 3)
# ys = np.array(ys)
# t = ys[:, 0] - ys[0, 0]
# axs[0, 0].plot(t, ys[:, 1], label='y')
# axs[0, 0].legend()
# axs[0, 1].plot(t, ys[:, 2], label='dy')
# axs[0, 1].legend()
# axs[0, 2].plot(t, ys[:, 3], label='ddy')
# axs[0, 2].legend()
# axs[1, 0].plot(t, ys[:, 4], label='zmp')
# axs[1, 0].legend()
# axs[1, 1].plot(t, ys[:, 5], label='zmpd')
# axs[1, 1].legend()
# plt.show()

# test generatedata
# data = generatedata()
# data = np.array(data)

# test _read_and_fit_data
data = np.loadtxt('data.csv', delimiter=',')
x = data[:, :3]
y1 = data[:, 3]
param = fitdata2(x, y1)
# plot
y1_fit = _fit_function2(x, *param[0])
plt.figure(figsize=(10, 6))
plt.plot(y1, label='Raw Data', color='blue')
plt.plot(y1_fit, label='Fitted Curve', color='red', linestyle='--')
plt.xlabel('Index')
plt.ylabel('Data')
plt.title('Comparison of Raw Data and Fitted Curve')
plt.legend()
plt.grid()
plt.show()
