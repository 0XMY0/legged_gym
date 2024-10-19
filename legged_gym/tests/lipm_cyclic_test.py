import numpy as np
from math import exp

class LIPM:
    def __init__(self, height, T, disy):
        self.height = height
        self.disy = 0.10
        self.g = 9.81
        self.w0 = (self.g/self.height)**0.5
        self.T = T
        self.dsupphase = 0.111111
        self.qy0 = self._getqy0_dsup()

    def _getbound(self):
        bounds = [0, 0]
        Kp = 10.0
        t = 0
        step = 0
        sup = 'R'
        y = [-0.025, 0.0] # pos vel
        y[1] = (-self.qy0 - y[0]) * self.w0
        ys = []
        for i in range(0, np.floor(6 * self.T / 0.004).astype(int)):
            t += 0.004
            if t > self.T:
                sup = 'R' if sup == 'L' else 'L'
                t -= self.T
                step += 1
            qy0 = self.qy0 if sup == 'L' else -self.qy0
            anky = self.disy if sup == 'L' else -self.disy
            qyd = (qy0 - anky) * np.exp(self.w0 * t) + anky
            qynow = y[0] + y[1] / self.w0
            if t < (1.0 - self.dsupphase) * self.T:
                zmpy = Kp * (qynow - qyd) + anky
            else:
                zmpy = Kp * (qynow - qyd) - anky
            ddy = self.g / self.height * (y[0] - zmpy)
            y[0] += y[1] * 0.004 + 0.5 * ddy * 0.004**2
            y[1] += ddy * 0.004
            if(step >= 4 and step <= 5):
                bounds[0] = min(bounds[0], y[1])
                bounds[1] = max(bounds[1], y[1])
            ys.append([y[0], y[1], ddy, zmpy])

        with open('lipm_cyclic_test.csv', 'w') as f:
            f.write('t,y,dy,ddy,zmpy\n')
            for y in ys:
                f.write(','.join(map(str, y)) + '\n')
        return (bounds[1] - bounds[0]) / 2.0
    
    def _getqy0_dsup(self) -> float:
        def _getqend(q0):
            q1 = (q0 - self.disy) * exp(self.w0 * self.T * (1 - self.dsupphase)) + self.disy
            q2 = (q1 - -self.disy) * exp(self.w0 * self.T * self.dsupphase) + -self.disy
            return q2
        q0 = 0.05
        while(1):
            qnow = q0
            qend = _getqend(q0)
            # min f = (qend + q0)^2
            qend_add = _getqend(q0 + 1e-5)
            qend_sub = _getqend(q0 - 1e-5)
            qend_grad = (qend_add - qend_sub) / 2e-5
            f_grad = 2 * (qend + q0) * (qend_grad + 1)
            q0 -= f_grad * 1e-3
            if(abs(f_grad) < 1e-5):
                break
        return q0

# data = []
# for i in range(0, 10000):
#     height_rand_bounds = [0.4, 0.8]
#     T_rand_bounds = [0.2, 1.0]
#     lipm = LIPM(np.random.uniform(height_rand_bounds[0], height_rand_bounds[1]), np.random.uniform(T_rand_bounds[0], T_rand_bounds[1]), 0.10)
#     data.append([lipm.height, 1.0 / (2 * lipm.T), lipm._getbound()])
# with open('lipm_cyclic_test.csv', 'w') as f:
#     f.write('height,T,disy\n')
#     for d in data:
#         f.write(','.join(map(str, d)) + '\n')

# fitted bound for vy from MATLAB: f(x, y) = 0.9761 - 1.106 * x - 0.2678 * y + 0.531 * x^2 + 0.06972 * x * y + 0.03481 * y^2, where x = height, y = frequency, f = vy bound

# test
# height_rand_bounds = [0.4, 0.8]
# T_rand_bounds = [0.2, 1.0]
# lipm1 = LIPM(np.random.uniform(height_rand_bounds[0], height_rand_bounds[1]), np.random.uniform(T_rand_bounds[0], T_rand_bounds[1]), 0.10)
# lipm2 = LIPM(0.55, 0.33, 0.10)

# x = lipm1.height
# y = 1.0 / (2 * lipm1.T)
# print('original bound:', lipm1._getbound())
# print('fitted bound:', 0.9761 - 1.106 * x - 0.2678 * y + 0.531 * x**2 + 0.06972 * x * y + 0.03481 * y**2)

lipm2 = LIPM(0.55, 0.3333333, 0.10)
print(lipm2._getbound())
