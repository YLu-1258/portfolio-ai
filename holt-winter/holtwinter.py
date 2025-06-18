import numpy as np


class AdditiveHoltWinters:
    def __init__(self, alpha: float, beta: float, gamma: float, obs_per_season: int):
        """
        alpha, beta, gamma: Smoothing parameters for Holt-Winters.
        obs_per_season: The number of observations that make up a season.

        e.g. monthly data with yearly seasonality --> obs_per_season = 12
        e.g. daily data with weekly seasonality --> obs_per_season = 7
        """

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.m = obs_per_season

        self._is_fit = False

    def fit(self, series: list):
        """
        Fits the additive Holt-Winters model using the smoothing parameters it was initialized with.
        """

        if self._is_fit:
            raise Exception("This model has already been fit!")

        y = np.array(series)
        N = len(y)
        m = self.m

        # Convenience
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        # Level, trend, season
        l = np.zeros(N)
        b = np.zeros(N)
        s = np.zeros(N)

        # Initialize the level, trend, and season by regression fit of a first-order
        # Fourier series, as suggested by Hyndman in the article below. The intution for
        # this lies in the fact that seasonality inherently implies periodicity. We
        # simplify by only considering the first harmonic

        # https://robjhyndman.com/hyndsight/hw-initialization/

        # l_0 + b_0*t + A*cos(2pi*t/m) + B*sin(2pi*t/m)
        X = np.column_stack(
            [
                np.ones(N),
                t := np.arange(N),
                np.cos(2 * np.pi * t / m),
                np.sin(2 * np.pi * t / m),
            ]
        )
        l[0], b[0], A, B = np.linalg.lstsq(X, y)[0]

        for i in range(m):
            s[i] = A * np.cos(2 * np.pi * i / m) + B * np.sin(2 * np.pi * i / m)

        # Ensure that the seasonal factors have no net change on the average level
        s[:m] -= s[:m].mean()

        for t in range(1, N):
            if t < m:
                l[t] = y[t]

                # Initialize by carrying forward the trend
                b[t] = b[t - 1]

                # Optionally, clobber the computed seasonal values. You may want to do this
                # if your initial seasonal estimates are unstable or the Fourier fit is poor
                # s[t] = s[t-1]  # To do so, uncomment this line
                continue

            # https://otexts.com/fpp2/holt-winters.html
            l[t] = alpha * (y[t] - s[t - m]) + (1 - alpha) * (l[t - 1] + b[t - 1])
            b[t] = beta * (l[t] - l[t - 2]) + (1 - beta) * b[t - 1]
            s[t] = gamma * (y[t] - l[t - 1] - b[t - 1]) + (1 - gamma) * s[t - m]

        self.N = N
        self.l = l
        self.b = b
        self.s = s

        self._is_fit = True

    def forecast(self, steps: int):
        """
        Forecasts steps steps into the future.
        """

        if not self._is_fit:
            raise Exception("This model must be fit before it can forecast!")

        f = np.zeros(steps)
        for h in range(1, steps + 1):
            # In essence, reference the equivalent value from last season
            # The textbook computes t + h - m(k + 1), where k = floor((h - 1)/m),
            # but this can easily be simplified to t - m + ((h - 1) mod m)
            i = self.N - self.m + ((h - 1) % self.m)
            f[h - 1] = self.l[-1] + h * self.b[-1] + self.s[i]

        return f


# For testing purposes, consider the following initialization for quarterly data:
# test = AdditiveHoltWinters(alpha=0.3, beta=0.1, gamma=0.1, obs_per_season=4)
