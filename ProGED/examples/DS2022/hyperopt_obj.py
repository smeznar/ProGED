import numpy as np
import ProGED as pg
from ProGED.examples.generate_data_ODE_systems import generate_ODE_data, lorenz, VDP


class Estimation:
    def __init__(self, system, inits=None):
        self.systems = {"lorenz": lorenz, "VDP": VDP}
        self.system = self.systems[system]

        self.inits = inits
        if system == "lorenz":
            if not inits:
                self.inits = [0, 1, 1.05]
            expr = ["C*(y-x)", "x*(C-z)-y", "x*y - C*z"]
            symbols = {"x": ["x", "y", "z"], "const": "C"}
            self.models = pg.ModelBox()
            self.models.add_model(expr, symbols)

        self.data = generate_ODE_data(system, self.inits)

    def fit(self, hyperparams):
        objective_settings = {
            "atol": 10 ** (-6),
            "rtol": 10 ** (-4),
            "max_step": 10 ** 3
        }
        optimizer_settings = {
            "f": hyperparams[0],
            "cr": hyperparams[1],
            "pop_size": hyperparams[2],
            "max_iter": hyperparams[3],
            "lower_upper_bounds": (-10, 10),
            "atol": 0.01,
            "tol": 0.01
        }
        estimation_settings = {
            "max_constants": 10,
            "optimizer_settings": optimizer_settings,
            "objective_settings": objective_settings,
            "verbosity": 0
        }

        try:
            models_fit = pg.fit_models(self.models, self.data, task_type='differential', time_index=0,
                            estimation_settings=estimation_settings)
        except Exception as e:
            print("Error in ", str(hyperparams))
            print(e)
            return 1e9
        
        return models_fit[0].get_error()

if __name__ == "__main__":
    est = Estimation("lorenz")
    print(est.models)

    res = est.fit([0.5, 0.9, 3, 2])
    print(res)