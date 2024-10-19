import numpy as np
class LIPM:
    def __init__(self, height, T, disy):
        self.height = height
        self.disy = 0.10
        self.g = 9.81
        self.w0 = (self.g/self.height)**0.5
        self.T = T
        self.qy0 = self.disy * (np.exp(self.w0 * self.T) - 1.0) / (np.exp(self.w0 * self.T) + 1.0)

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
            zmpy = Kp * (qynow - qyd) + anky
            ddy = self.g / self.height * (y[0] - zmpy)
            y[0] += y[1] * 0.004 + 0.5 * ddy * 0.004**2
            y[1] += ddy * 0.004
            if(step >= 4 and step <= 5):
                bounds[0] = min(bounds[0], y[1])
                bounds[1] = max(bounds[1], y[1])

        return (bounds[1] - bounds[0]) / 2.0

data = []
for i in range(0, 1000):
    height_rand_bounds = [0.4, 0.8]
    T_rand_bounds = [0.2, 1.2]
    lipm = LIPM(np.random.uniform(height_rand_bounds[0], height_rand_bounds[1]), np.random.uniform(T_rand_bounds[0], T_rand_bounds[1]), 0.10)
    data.append([lipm.height, 1.0 / lipm.T, lipm._getbound()])

# # print data to csv
# with open('lipm_cyclic_test.csv', 'w') as f:
#     f.write('height,T,disy\n')
#     for d in data:
#         f.write(','.join(map(str, d)) + '\n')

# fitted: bound = -0.3936 * height - 0.06757 * freq + 0.6992
# test
height_rand_bounds = [0.4, 0.8]
T_rand_bounds = [0.2, 1.2]
lipm = LIPM(np.random.uniform(height_rand_bounds[0], height_rand_bounds[1]), np.random.uniform(T_rand_bounds[0], T_rand_bounds[1]), 0.10)
print('original bound:', lipm._getbound())
print('fitted bound:', -0.3936 * 0.6 - 0.06757 * 1.0 + 0.6992)
