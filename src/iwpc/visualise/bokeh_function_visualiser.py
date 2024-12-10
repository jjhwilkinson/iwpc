import datetime as dt
from abc import abstractmethod, ABC
from collections import OrderedDict
from typing import List, Optional, Callable, Tuple

import numpy as np
from bokeh.io import curdoc
from bokeh.models import (
    Column, Select, Slider, Switch, Row, PreText, Div, Button, Spinner, CustomJS,
)
from numpy import ndarray

from iwpc.scalars.scalar import Scalar
from iwpc.scalars.scalar_function import ScalarFunction


class BokehFunctionVisualiser(ABC):
    """
    Base class for a function visualiser implemented in bokeh that allows for rich interactive web-browser based plots
    that can be hosted on a server and shared with others
    """
    def __init__(
        self,
        fn: Callable[[ndarray], ndarray],
        input_scalars: List[Scalar],
        output_scalars: List[ScalarFunction],
        center_point: Optional[ndarray] = None,
        initial_output_scalar_ind: int = 0,
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
        center_point
            The default value to use for each input scalar. Defaults to the middle of the bins attribute of each scalar
        initial_output_scalar_ind
            The index of the output scalar to initially select on start up
        """
        self.function = fn
        self.input_scalars = input_scalars
        self.output_scalars = output_scalars
        self.center_point = center_point
        self.initial_output_scalar_ind = initial_output_scalar_ind
        if center_point is None:
            self.center_point = np.asarray([scalar.bins[len(scalar.bins) // 2] for scalar in input_scalars])

        self.input_scalar_menu = OrderedDict([(scalar.label, scalar) for scalar in self.input_scalars])
        self.output_scalar_menu = OrderedDict([(scalar.label, scalar) for scalar in self.output_scalars])

        self.main_figure = None
        self.last_scalar_output = None
        self.input_pickers = []
        self.setup()
        self.update_output()

    @abstractmethod
    def setup_figure(self) -> None:
        """
        Abstract method to define the primary figure of the Visualiser
        """
        pass

    def setup_settings_column(self) -> None:
        """
        Sets up the right hand column of settings including the scalar selectors, sliders, and more
        """
        self.output_scalar_picker = Select(
            title="Output Scalar",
            options=list(self.output_scalar_menu.keys()),
            sizing_mode='scale_width',
            value=self.output_scalars[self.initial_output_scalar_ind].label,
        )
        self.min_output = Spinner(value=0., sizing_mode='stretch_width', title='Output range min')
        self.max_output = Spinner(value=1., sizing_mode='stretch_width', title='Output range max')
        for widget in [self.output_scalar_picker, self.min_output, self.max_output]:
            widget.on_change(
                'value',
                lambda attr, old, new: self.update_output(reuse_previous_output=True)
            )

        self.use_custom_output_range = Switch(active=False)
        self.use_custom_output_range.on_change(
            'active',
            lambda attr, old, new: self.update_output(reuse_previous_output=True)
        )

        self.sliders = [Slider(
            start=scalar.bins[0],
            end=scalar.bins[-1],
            value=self.center_point[i],
            step=(scalar.bins[1] - scalar.bins[0]),
            title=scalar.latex_label,
            sizing_mode='stretch_width',
        ) for i, scalar in enumerate(self.input_scalars)]
        for s in self.sliders:
            s.on_change('value_throttled', lambda attr, old, new: self.update_output())

        self.freeze_input_axes_switch = Switch(active=False)
        self.freeze_output_axes_switch = Switch(active=False)
        self.axis_resolutions = [
            Spinner(low=2, step=1, value=100, width=80, sizing_mode='stretch_height', title='Num points') for _ in
            self.input_pickers
        ]
        for s in self.axis_resolutions:
            s.on_change('value', lambda attr, old, new: self.update_output())

        self.reset_button = Button(label="Reset")
        self.reset_button.on_click(self.reset_sliders)
        self.last_updated_div = Div(text="")

        self.settings_column = Column(
            *[Row(picker, res, sizing_mode='stretch_width') for picker, res in
              zip(self.input_pickers, self.axis_resolutions)],
            self.output_scalar_picker,
            Row(
                self.min_output,
                self.max_output,
                Row(
                    PreText(text="Use: "),
                    self.use_custom_output_range,
                    sizing_mode='stretch_height'
                ),
                sizing_mode='stretch_width'
            ),
            Row(PreText(text="Freeze input axes auto-scale"), self.freeze_input_axes_switch),
            Row(PreText(text="Freeze output axis auto-scale"), self.freeze_output_axes_switch),
            Div(text="<h2><b>Input Sliders</b></h2>", sizing_mode='stretch_width'),
            *self.sliders,
            self.reset_button,
            self.last_updated_div,
            sizing_mode='stretch_height',
            width=300,
        )

    @abstractmethod
    def setup(self) -> None:
        """
        Abstract method to define and configure all widgets needed by the UI
        """
        self.setup_figure()
        self.setup_input_scalar_pickers()
        self.setup_settings_column()

        for picker in self.input_pickers:
            picker.on_change('value', lambda attr, old, new: self.update_output())

    @abstractmethod
    def update_input_axes(self) -> None:
        """
        Abstract method to update the labels and ranges of all the axes corresponding to the input scalars
        """
        pass

    @abstractmethod
    def update_output_axes(self) -> None:
        """
        Abstract method to update the labels and ranges of all the axes corresponding to the output scalar
        """
        pass

    @abstractmethod
    def _update_output(self, reuse_previous_output: bool = False) -> None:
        """
        Abstract method to update the output of the function. Derived visualisers must define this

        Parameters
        ----------
        reuse_previous_output
            Whether to reuse the outputs of the previous function evaluation. Useful if the user has performed some
            action that would not affect the output of the function such as changing the output scalar
        """

    def update_output(self, reuse_previous_output: bool = False) -> None:
        """
        Updates the output of the function and the plots

        Parameters
        ----------
        reuse_previous_output
            Whether to reuse the outputs of the previous function evaluation. Useful if the user has performed some
            action that would not affect the output of the function such as changing the output scalar
        """
        self._update_output(reuse_previous_output=reuse_previous_output)
        self.update_input_axes()
        self.update_output_axes()
        curdoc().add_next_tick_callback(self.update_last_updated)

    def update_last_updated(self) -> None:
        """
        Updates the date and time of the 'last updated' label
        """
        self.last_updated_div.text = f"<p>Last updated: {dt.datetime.now().strftime('%y-%m-%d %H:%M:%S')}</p>"

    @property
    def input_scalar_ind1(self) -> int:
        """
        Returns the index of the first input scalar in self.input_scalars
        """
        return list(self.input_scalar_menu.keys()).index(self.input_pickers[0].value)

    @property
    def output_scalar_ind(self) -> int:
        """
        Returns the index of the output scalar in self.output_scalars
        """
        return list(self.output_scalar_menu.keys()).index(self.output_scalar_picker.value)

    @property
    def input_scalar1(self) -> Scalar:
        """
        Returns
        -------
        Scalar
            The first selected input scalar
        """
        return self.input_scalars[self.input_scalar_ind1]

    @property
    def xbins(self) -> ndarray:
        """
        Returns
        -------
        ndarray
            An array containing the values of the first selected input scalar at which the function should be evaluated
        """
        return np.linspace(self.input_scalar1.bins[0], self.input_scalar1.bins[-1], int(self.axis_resolutions[0].value))

    @abstractmethod
    def setup_input_scalar_pickers(self) -> None:
        """
        Method to define the Select widgets for the input scalars must place the constructed widgets into
        self.input_pickers
        """

    @property
    def output_scalar(self) -> ScalarFunction:
        """
        Returns
        -------
        Scalar
            The selected output scalar
        """
        return self.output_scalars[self.output_scalar_ind]

    def output_scalar_range(self, output_values: ndarray) -> Tuple[float, float]:
        """
        Calculates the range of the output range for adjusting axes. If the custom output range stich is active returns
        the values of the custom range inputs. Otherwise, returns the min and max values of the self.output_scalar.bins
        if provided. Otherwise, returns a range 10% larger either side of the min/max of output_values

        Returns
        -------
        Tuple[float, float]
            The min and max values of the output_scalar's range for adjusting axes
        """
        if self.use_custom_output_range.active:
            return self.min_output.value, self.max_output.value

        if self.output_scalar.bins is not None:
            return min(self.output_scalar.bins), max(self.output_scalar.bins)

        output_values = output_values[np.isfinite(output_values)]
        output_range = output_values.max() - output_values.min()
        if output_range == 0:
            output_range = 1
        return output_values.min() - 0.1 * output_range, output_values.max() + 0.1 * output_range

    def reset_sliders(self) -> None:
        """
        Resets the value of each slider to its corresponding value in self.center_point
        """
        for i, slider in enumerate(self.sliders):
            slider.value = self.center_point[i]
        self.update_output()
