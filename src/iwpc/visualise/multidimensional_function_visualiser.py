from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Optional, Tuple, Callable, Any

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from numpy._typing import NDArray

from iwpc.scalars.scalar import Scalar
from iwpc.scalars.scalar_function import ScalarFunction


class MultidimensionalFunctionVisualiser(ABC):
    """
    Abstract base-class and GUI skeleton for a multidimensional function plotter. Implementations are useful for
    plotting low dimensional slices of a multidimensional function's output.

    WARNING, when using the visualiser, you must assign the instance to a variable. If not, the object may go out of
    scope and the garbage collector will delete it from memory causing a freeze.

    Good:
    x = MultidimensionalFunctionVisualiser(...)
    plt.show()

    Bad:
    MultidimensionalFunctionVisualiser(...)
    plt.show()
    """
    def __init__(
        self,
        fn: Callable[[NDArray], NDArray],
        input_scalars: List[Scalar],
        output_scalars: List[ScalarFunction],
        num_plot_points: int = 100,
        label_width: float = 0.15,
        label_height: float = 0.03,
        center_point: Optional[List[float]] = None,
        fig_size: Tuple[float, float] = (12, 8),
    ):
        """
        Parameters
        ----------
        fn
            The function to be plotted
        input_scalars
            A list of Scalar objects describing the input features of the function. These are used to provide axis
            labels and sliders for conditioning values
        output_scalars
            A list of ScalarFunction objects describing plottable output features of the function and how to obtain them
            from the function's actual output. These are used to provide axis labels and obtain the plotted values
        num_plot_points
            The number of points to divide input feature axes into
        label_width
            The width to use for components that render the Scalar labels. The default value is pretty reasonable but
            one may want to increase this if the labels are long
        label_height
            The height to use for components that render the Scalar labels. The default value is pretty reasonable but
            one may want to increase this if the labels are tall (like LaTeX fractions)
        center_point
            The default value to use for each input scalar. Defaults to the middle of the bins attribute of each scalar
        fig_size
            The size of the interactive plot figure
        """
        assert len(input_scalars) > 0 and len(output_scalars) > 0
        assert center_point is None or len(center_point) == len(input_scalars)
        self.function = fn
        self.input_scalars = input_scalars
        self.output_scalars = output_scalars
        self.num_plot_points = num_plot_points
        self.label_width = label_width
        self.label_height = label_height
        self.center_point = center_point if center_point is not None else [(inp.bins[0] + inp.bins[-1]) / 2 for inp in self.input_scalars]
        self.fig_size = fig_size

        self.fig = None
        self.ax = None
        self.sliders = None
        self.button = None
        self.setup()

    @property
    @abstractmethod
    def plot_dimension(self):
        """
        Returns
        -------
        int
            The number of dimensions of the multidimensional function
        """

    @property
    def input_scalar_labels(self) -> List[str]:
        """
        Returns
        -------
        List[str]
            The labels of each of the input scalars
        """
        return [scalar.latex_label for scalar in self.input_scalars]

    @property
    def output_scalar_labels(self) -> List[str]:
        """
        Returns
        -------
        List[str]
            The labels of each of the output scalars
        """
        return [scalar.latex_label for scalar in self.output_scalars]

    @property
    def num_input_scalars(self) -> int:
        """
        Returns
        -------
        int
            The number of input scalars
        """
        return len(self.input_scalars)

    @property
    def num_output_scalars(self) -> int:
        """
        Returns
        -------
        int
            The number of output scalars
        """
        return len(self.output_scalars)

    @property
    def max_in_out_scalars(self) -> int:
        """
        Returns
        -------
        int
            The larger of the number of input scalars and the number of output scalars
        """
        return max(self.num_output_scalars, self.num_input_scalars)

    @property
    def bottom_height(self) -> float:
        """
        Returns
        -------
        float
            The height of the bottom section of the interactive plot containing the sliders and radio buttons
        """
        return self.max_in_out_scalars * self.label_height

    @property
    def radio_width(self) -> float:
        """
        The width of the radio button section for selecting the plotted variables
        """
        return (self.plot_dimension + 1) * self.label_width + 0.02

    @abstractmethod
    def setup_radio_buttons(self) -> None:
        """
        Sets up the radio buttons for selecting the scalars shown in the plot
        """

    @abstractmethod
    def update_axes(self) -> None:
        """
        Configures the axis labels and limits of the plot
        """

    @abstractmethod
    def setup_plot(self) -> None:
        """
        Configures the plot
        """

    @abstractmethod
    def update_plot(self) -> None:
        """
        Updates the values in the plot
        """

    def reset(self, event: Any) -> None:
        """
        Callback function for the reset button. Resets the sliders to their default values
        """
        for slider in self.sliders.values():
            slider.reset()

    def update(self, event: Any) -> None:
        """
        Callback function that updates the values in the plot as well as the axes and redraws the figure
        """
        self.update_axes()
        self.update_plot()
        self.fig.canvas.draw_idle()

    def setup_sliders(self):
        """
        Sets up the sliders used for setting input scalar values
        """
        self.sliders = OrderedDict()
        slider_spacing = self.bottom_height / (len(self.input_scalars) + 1)
        for i, (init, scalar) in enumerate(list(zip(self.center_point, self.input_scalars))):
            slider_ax = plt.axes(
                (
                    self.radio_width,
                    self.bottom_height - slider_spacing * (i + 1) - self.label_height / 2,
                    1 - self.radio_width - 0.08,
                    self.label_height,
                )
            )

            self.sliders[scalar.label] = Slider(
                slider_ax,
                '',
                scalar.bins[0],
                scalar.bins[-1],
                valinit=init,

            )
            self.sliders[scalar.label].on_changed(self.update)

    def setup_reset_button(self) -> None:
        """
        Sets up the reset button
        """
        reset_ax = plt.axes((self.radio_width, self.bottom_height, 0.075, 0.03))
        self.button = Button(reset_ax, 'Reset', hovercolor='0.975')
        self.button.on_clicked(self.reset)

    def setup(self) -> None:
        """
        Sets up the figure
        """
        self.fig, self.ax = plt.subplots(figsize=self.fig_size)
        plt.subplots_adjust(left=0.08, bottom=self.bottom_height + 0.1, top=0.97, right=0.97)

        self.setup_radio_buttons()
        self.setup_sliders()
        self.setup_reset_button()
        self.setup_plot()
        self.update(None)
