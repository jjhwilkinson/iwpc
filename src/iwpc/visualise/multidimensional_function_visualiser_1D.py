import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons
from numpy._typing import NDArray

from iwpc.scalars.scalar import Scalar
from iwpc.scalars.scalar_function import ScalarFunction
from iwpc.visualise.multidimensional_function_visualiser import MultidimensionalFunctionVisualiser


class MultidimensionalFunctionVisualiser1D(MultidimensionalFunctionVisualiser):
    """
    MultidimensionalFunctionVisualiser implementation for plotting 1-dimensional slices of a function's output using a
    line-plot

    WARNING, when using the visualiser, you must assign the instance to a variable. If not, the object may go out of
    scope and the garbage collector will delete it from memory causing a freeze.

    Good:
    x = MultidimensionalFunctionVisualiser(...)
    plt.show()

    Bad:
    MultidimensionalFunctionVisualiser(...)
    plt.show()
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        args
            Any positional arguments accepted by MultidimensionalFunctionVisualiser
        kwargs
            Any keyword arguments accepted by MultidimensionalFunctionVisualiser
        """
        self.line = None
        self.x_axis_variable_radio = None
        self.y_axis_variable_radio = None

        super().__init__(*args, **kwargs)

    @property
    def plot_dimension(self) -> int:
        return 1

    def setup_radio_buttons(self) -> None:
        """
        Sets up a set of radio buttons for the x-axis variable and the y-axis variable
        """
        y_axis_rax = plt.axes((0., 0, self.label_width, self.bottom_height))
        y_axis_rax.set_title("y-axis")
        self.y_axis_variable_radio = RadioButtons(
            y_axis_rax,
            tuple(self.output_scalar_labels),
            active=0,
        )
        self.y_axis_variable_radio.on_clicked(self.update)

        x_axis_rax = plt.axes((self.label_width, 0, self.label_width, self.bottom_height))
        x_axis_rax.set_title("x-axis")
        self.x_axis_variable_radio = RadioButtons(
            x_axis_rax,
            tuple(self.input_scalar_labels),
            active=0,
        )
        self.x_axis_variable_radio.on_clicked(self.update)

    def update_axes(self) -> None:
        """
        Sets the label and limits of the x and y axes
        """
        self.update_x_axis()
        self.update_y_axis()

    def setup_plot(self) -> None:
        """
        Creates the initial line plot
        """
        self.line, = self.ax.plot(self.x_axis_points, self.y_axis_scalar(self.evaluate_function()))

    def update_plot(self) -> None:
        """
        Updates the line plot with the current selected x and y axes values
        """
        self.line.set_data(self.x_axis_points, self.y_axis_scalar(self.evaluate_function()))

    @property
    def x_axis_scalar(self) -> Scalar:
        """
        Returns
        -------
        Scalar
            The current selected input scalar for the x-axis
        """
        return self.input_scalars[self.x_axis_variable_radio.index_selected]

    @property
    def x_axis_points(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            An array of values of the current selected input scalar for the x-axis
        """
        return np.linspace(self.x_axis_scalar.bins[0], self.x_axis_scalar.bins[-1], self.num_plot_points)

    @property
    def y_axis_scalar(self) -> ScalarFunction:
        """
        Returns
        -------
        ScalarFunction
            The current selected output scalar for the y-axis
        """
        return self.output_scalars[self.y_axis_variable_radio.index_selected]

    def update_x_axis(self) -> None:
        """
        Sets the label and limits of the x-axis
        """
        self.ax.set_xlabel(self.x_axis_scalar.latex_label)
        self.ax.set_xlim((self.x_axis_scalar.bins[0], self.x_axis_scalar.bins[-1]))

    def update_y_axis(self):
        """
        Sets the label and limits of the y-axis
        """
        self.ax.set_ylabel(self.y_axis_scalar.latex_label)
        if self.y_axis_scalar.bins is not None:
            y_min, y_max = self.y_axis_scalar.bins[0], self.y_axis_scalar.bins[-1]
        else:
            y_values = self.evaluate_y_values()
            y_values = y_values[np.isfinite(y_values)]
            y_range = y_values.max() - y_values.min()
            if y_range == 0:
                y_range = 1
            y_min, y_max = y_values.min() - 0.1 * y_range, y_values.max() + 0.1 * y_range

        self.ax.set_ylim((y_min, y_max))

    def evaluate_function(self) -> NDArray:
        """
        Evaluates the function on the range of x-axis values for the selected input scalar and fixed slider values

        Returns
        -------
        NDArray
            The output of the function for the given slider values and across the x-axis scalar
        """
        eval_point = np.asarray(list(slider.val for slider in self.sliders.values()))
        input = np.tile(eval_point, (self.num_plot_points, 1))
        input[:, self.x_axis_variable_radio.index_selected] = self.x_axis_points

        return self.function(input)

    def evaluate_y_values(self) -> NDArray:
        """
        Evaluates the selected output scalar function on the output of self.fn for the range of x-axis values of the
        selected input scalar and fixed slider values

        Returns
        -------
        NDArray
            The output value of the selected output scalar
        """
        return self.y_axis_scalar(self.evaluate_function())
