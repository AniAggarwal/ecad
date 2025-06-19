from pymoo.core.sampling import Sampling
import numpy as np
import numpy.typing as npt


class BinaryRandomSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs) -> npt.NDArray[np.bool_]:
        # Create a population of binary vectors (0s or 1s) of shape (n_samples, n_var)
        return np.random.randint(0, 2, size=(n_samples, problem.n_var)).astype(
            np.bool_
        )
