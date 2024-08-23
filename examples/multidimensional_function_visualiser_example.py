import numpy as np
from matplotlib import pyplot as plt

from iwpc.scalars.scalar import Scalar
from iwpc.scalars.scalar_function import ScalarFunction
from iwpc.visualise.multidimensional_function_visualiser_1D import MultidimensionalFunctionVisualiser1D
from iwpc.visualise.multidimensional_function_visualiser_2D import MultidimensionalFunctionVisualiser2D


def multidimensional_sinc(x):
    r = np.sqrt(np.sum(x**2, axis=1))
    return np.sin(r) / r


if __name__ == "__main__":
    num_input_scalars = 3

    input_scalars = [Scalar(f'x{i}', f'$x_{i}$', bins=np.linspace(-10, 10, 100)) for i in range(num_input_scalars)]
    output_scalars = [ScalarFunction(lambda x: x, f'sin(r) / r', r'$\frac{\sin(r)}{r}$')]

    visualiser1D = MultidimensionalFunctionVisualiser1D(
        multidimensional_sinc,
        input_scalars,
        output_scalars,
    )

    visualiser2D = MultidimensionalFunctionVisualiser2D(
        multidimensional_sinc,
        input_scalars,
        output_scalars,
    )
    plt.show()
