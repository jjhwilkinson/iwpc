import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons
from numpy._typing import NDArray

from iwpc.scalars.scalar import Scalar
from iwpc.scalars.scalar_function import ScalarFunction
from iwpc.visualise.multidimensional_function_visualiser import MultidimensionalFunctionVisualiser


class MultidimensionalFunctionVisualiser2D(MultidimensionalFunctionVisualiser):
    """
    MultidimensionalFunctionVisualiser implementation for plotting 2-dimensional slices of a function using a heat map

    WARNING, when using the visualiser, you must assign the instance to a variable. If not, the object may go out of
    scope and the garbage collector will delete it from memory causing a freeze.

    Good:
    x = MultidimensionalFunctionVisualiser(...)
    plt.show()

    Bad:
    MultidimensionalFunctionVisualiser(...)
    plt.show()
    """

    def __init__(self, *args, cmap: str = 'viridis', **kwargs) -> None:
        """
        Parameters
        ----------
        args
            Any positional arguments accepted by MultidimensionalFunctionVisualiser
        cmap
            The colormap to use for the heatmap
        kwargs
            Any keyword arguments accepted by MultidimensionalFunctionVisualiser
        """
        self.image = None
        self.c_bar = None
        self.x_axis_variable_radio = None
        self.y_axis_variable_radio = None
        self.z_axis_variable_radio = None
        self.cmap = cmap

        super().__init__(*args, **kwargs)

    @property
    def plot_dimension(self) -> int:
        return 2

    def setup_radio_buttons(self) -> None:
        """
        Sets up a set of radio buttons for the x-axis variable, y-axis variable, and z-axis variable
        """
        z_axis_rax = plt.axes((0, 0, self.label_width, self.bottom_height))
        z_axis_rax.set_title("z-axis")
        self.z_axis_variable_radio = RadioButtons(
            z_axis_rax,
            tuple(self.output_scalar_labels),
            active=0,
        )
        self.z_axis_variable_radio.on_clicked(self.update)

        x_axis_rax = plt.axes((self.label_width, 0, self.label_width, self.bottom_height))
        x_axis_rax.set_title("x-axis")
        self.x_axis_variable_radio = RadioButtons(
            x_axis_rax,
            tuple(self.input_scalar_labels),
            active=0,
        )
        self.x_axis_variable_radio.on_clicked(self.update)

        y_axis_rax = plt.axes((2 * self.label_width, 0, self.label_width, self.bottom_height))
        y_axis_rax.set_title("y-axis")
        self.y_axis_variable_radio = RadioButtons(
            y_axis_rax,
            tuple(self.input_scalar_labels),
            active=1,
        )
        self.y_axis_variable_radio.on_clicked(self.update)

    def update_axes(self) -> None:
        """
        Sets the label and limits of the x and y axes and the label and colour scale of the z axis
        """
        self.update_x_axis()
        self.update_y_axis()
        self.update_z_axis()

    def setup_plot(self) -> None:
        """
        Sets up the initial heatmap
        """
        self.image = self.ax.imshow(self.evaluate_z_values(), origin='lower', aspect='auto', cmap=self.cmap, interpolation='none')
        self.c_bar = plt.colorbar(self.image)

    def update_plot(self) -> None:
        """
        Updates the heatmap to reflect the output scalar as a function of the current selected input scalars
        """
        self.image.set_array(self.evaluate_z_values())
        self.image.set_extent((self.x_axis_scalar.bins[0], self.x_axis_scalar.bins[-1], self.y_axis_scalar.bins[0], self.y_axis_scalar.bins[-1]))

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
    def y_axis_scalar(self) -> Scalar:
        """
        Returns
        -------
        Scalar
            The current selected input scalar for the y-axis
        """
        return self.input_scalars[self.y_axis_variable_radio.index_selected]

    @property
    def z_axis_scalar(self) -> ScalarFunction:
        """
        Returns
        -------
        ScalarFunction
            The current selected output scalar for the z-axis
        """
        return self.output_scalars[self.z_axis_variable_radio.index_selected]

    def update_x_axis(self) -> None:
        """
        Sets the label and limits of the x-axis
        """
        self.ax.set_xlabel(self.x_axis_scalar.latex_label)
        self.ax.set_xlim((self.x_axis_scalar.bins[0], self.x_axis_scalar.bins[-1]))

    def update_y_axis(self) -> None:
        """
        Sets the label and limits of the y-axis
        """
        self.ax.set_ylabel(self.y_axis_scalar.latex_label)
        self.ax.set_ylim((self.y_axis_scalar.bins[0], self.y_axis_scalar.bins[-1]))

    def update_z_axis(self) -> None:
        """
        Sets the label and color scale of the z-axis
        """
        self.c_bar.set_label(self.z_axis_scalar.latex_label)
        if self.z_axis_scalar.bins is not None:
            z_min, z_max = self.z_axis_scalar.bins[0], self.z_axis_scalar.bins[-1]
        else:
            z_values = self.evaluate_z_values()
            z_range = z_values.max() - z_values.min()
            if z_range == 0:
                z_range = 1
            z_min, z_max = z_values.min() - 0.1 * z_range, z_values.max() + 0.1 * z_range

        self.image.set_clim(z_min, z_max)

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
    def y_axis_points(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            An array of values of the current selected input scalar for the y-axis
        """
        return np.linspace(self.y_axis_scalar.bins[0], self.y_axis_scalar.bins[-1], self.num_plot_points)

    def evaluate_function(self) -> NDArray:
        """
        Evaluates the function on the range of x-axis and y-axis values for the selected input scalars and fixed slider
        values

        Returns
        -------
        NDArray
            The output of the function for the given slider values and across the x-axis scalar and y-axis scalar values
        """
        eval_point = np.asarray(list(slider.val for slider in self.sliders.values()))
        input = np.tile(eval_point, (self.num_plot_points ** 2, 1))
        X, Y = np.meshgrid(self.x_axis_points, self.y_axis_points)
        input[:, self.x_axis_variable_radio.index_selected] = X.flatten()
        input[:, self.y_axis_variable_radio.index_selected] = Y.flatten()
        output = self.function(input)

        return output.reshape((self.num_plot_points, self.num_plot_points) + output.shape[1:])

    def evaluate_z_values(self) -> NDArray:
        """
        Evaluates the selected  output scalar function on the output of self.fn for the range of x-axis and y-axis
        values of the selected input scalar and fixed slider values

        Returns
        -------
        NDArray
            The output value of the selected output scalar
        """
        return self.z_axis_scalar(self.evaluate_function())
