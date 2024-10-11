import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import HoverTool, LinearColorMapper, Span, ColorBar, Row, TabPanel, Tabs, Select
from bokeh.palettes import Viridis256
from bokeh.plotting import figure
from numpy import ndarray

from ..scalars.scalar import Scalar
from .bokeh_function_visualiser import BokehFunctionVisualiser
from .bokeh_function_visualiser_1D import BokehFunctionVisualiser1D


class BokehFunctionVisualiser2D(BokehFunctionVisualiser):
    """
    A 2D implementation of BokehFunctionVisualiser.
    """
    def setup_figure(self):
        """
        Configures a central heatmap figure for a 2D visualisation of the function as well as two additional
        line-profile cross-section figures
        """
        hover = HoverTool(
            tooltips=[
                ("x", "$x"),
                ("y", "$y"),
                ("value", "@image"),
            ]
        )
        self.main_figure = figure(
            x_range=(0, 1),
            y_range=(0, 1),
            sizing_mode='scale_both',
            width=100,
            height=100,
            x_axis_label='x',
            y_axis_label='y',
        )
        self.main_figure.add_tools(hover)
        self.main_figure.on_event('tap', self.handle_main_figure_click)
        self.color_mapper = LinearColorMapper(palette=Viridis256, low=-1, high=1)
        self.x_span = Span(
            location=0, dimension='height', line_color='red', line_width=1, line_dash='dashed', visible=False,
            line_alpha=0.5
        )
        self.main_figure.add_layout(self.x_span)
        self.y_span = Span(
            location=0, dimension='width', line_color='red', line_width=1, line_dash='dashed', visible=False,
            line_alpha=0.5
        )
        self.main_figure.add_layout(self.y_span)
        self.image = self.main_figure.image(
            image=[], x=0, y=0, dw=1, dh=1, visible=True, color_mapper=self.color_mapper, level="image"
        )
        self.colorbar = ColorBar(color_mapper=self.color_mapper, title='')
        self.main_figure.add_layout(self.colorbar, 'right')

        self.x_line_profile_figure = figure(
            x_range=self.main_figure.x_range, y_range=(self.color_mapper.low, self.color_mapper.high),
            sizing_mode='stretch_width', height=200
        )
        self.x_line_profile = self.x_line_profile_figure.line()
        self.y_line_profile_figure = figure(
            y_range=self.main_figure.y_range, x_range=(self.color_mapper.low, self.color_mapper.high),
            sizing_mode='stretch_height', width=200
        )
        self.y_line_profile = self.y_line_profile_figure.line()

        self.figure = gridplot(
            [[self.main_figure, self.y_line_profile_figure], [self.x_line_profile_figure, None]],
            sizing_mode='stretch_both'
        )

    def setup(self) -> None:
        """
        Configures the UI and provides a root 'tabs' widget, one for a 2D view of the function and one for a 1D view of
        the function
        """
        super().setup()

        self.visualiser_1d = BokehFunctionVisualiser1D(
            self.function, self.input_scalars, self.output_scalars, self.center_point
        )
        self.panel_2d = Row(self.figure, self.settings_column, sizing_mode='stretch_both')
        self.tab_1d = TabPanel(child=self.visualiser_1d.root, title="1D View")
        self.tab_2d = TabPanel(child=self.panel_2d, title="2D View")
        self.root = Tabs(tabs=[self.tab_2d, self.tab_1d], sizing_mode='stretch_both')

    @property
    def input_scalar_ind2(self) -> int:
        """
        Returns
        -------
        int
            The index of the second input scalar in self.input_scalars
        """
        return list(self.input_scalar_menu.keys()).index(self.input_pickers[1].value)

    @property
    def input_scalar2(self) -> Scalar:
        """
        Returns
        -------
        Scalar
            The second input scalar
        """
        return self.input_scalars[self.input_scalar_ind2]

    @property
    def ybins(self) -> ndarray:
        """
        Returns
        -------
        ndarray
            An array containing the values of the second selected input scalar at which the function should be evaluated
        """
        return np.linspace(self.input_scalar2.bins[0], self.input_scalar2.bins[-1], int(self.axis_resolutions[1].value))

    def update_input_axes(self) -> None:
        """
        Updates the label and range of the plot's x-axis and y-axis
        """
        self.main_figure.xaxis.axis_label = self.input_scalar1.latex_label
        self.main_figure.yaxis.axis_label = self.input_scalar2.latex_label
        self.colorbar.update(title=self.output_scalar.latex_label)
        self.x_line_profile_figure.yaxis.axis_label = self.output_scalar.latex_label
        self.y_line_profile_figure.xaxis.axis_label = self.output_scalar.latex_label

        if not self.freeze_input_axes_switch.active:
            self.main_figure.x_range.update(start=self.xbins[0], end=self.xbins[-1])
            self.main_figure.y_range.update(start=self.ybins[0], end=self.ybins[-1])

    def update_output_axes(self) -> None:
        """
        Updates the label and range of the plot's z-axis
        """
        z_min, z_max = self.output_scalar_range(self.last_output)
        self.x_line_profile_figure.y_range.update(start=z_min, end=z_max)
        self.y_line_profile_figure.x_range.update(start=z_min, end=z_max)
        self.color_mapper.update(low=z_min, high=z_max)

    def update_output(self) -> None:
        """
        Re-computes the output of the function and updates the data in the heatmap and line-profile cross sections
        """
        eval_point = np.asarray(list(slider.value for slider in self.sliders))
        input = np.tile(eval_point, (self.xbins.shape[0], self.ybins.shape[0], 1))
        input[..., self.input_scalar_ind1] = self.xbins[:, np.newaxis]
        input[..., self.input_scalar_ind2] = self.ybins[np.newaxis, :]
        input = input.reshape((-1, eval_point.shape[0]))
        self.last_output = self.output_scalar(self.function(input)).reshape((self.xbins.shape[0], self.ybins.shape[0]))
        self.image.data_source.data = {'image': [self.last_output.T]}
        delta_x = (self.xbins[1] - self.xbins[0])
        delta_y = (self.ybins[1] - self.ybins[0])
        self.image.glyph.update(
            x=self.xbins[0] - delta_x / 2,
            y=self.ybins[0] - delta_y / 2,
            dw=self.xbins[-1] - self.xbins[0] + delta_x,
            dh=self.ybins[-1] - self.ybins[0] + delta_y,
        )
        self.update_line_profiles()
        super().update_output()

    def setup_input_scalar_pickers(self) -> None:
        """
        Defines Select widgets for the x-axis and y-axis scalars
        """
        self.input_pickers = [
            Select(
                title="x-axis", options=list(self.input_scalar_menu.keys()), sizing_mode='scale_width',
                value=self.input_scalars[0].label
            ),
            Select(
                title="y-axis", options=list(self.input_scalar_menu.keys()), sizing_mode='scale_width',
                value=self.input_scalars[min(1, len(self.input_scalars) - 1)].label
            )
        ]
        for picker in self.input_pickers:
            picker.on_change('value', lambda attr, old, new: self.update_all())

    def nearest_bin_index(self, bins: ndarray, value: float) -> int:
        """
        Returns
        -------
        int
            The index of the bin in 'bins' with the value closest to 'value'
        """
        return np.abs(bins - value).argmin()

    def configure_1d_panel(self, selected_scalar_ind: int, selected_scalar_res: int, slider_values: ndarray) -> None:
        """
        Configures the settings of the 1D visualiser tab to show the specified visualisation

        Parameters
        ----------
        selected_scalar_ind
            The index of the x-axis scalar to select in self.visualiser_1d.input_scalars
        selected_scalar_res
            The number of points in the input scalar range on which to evaluate the function
        slider_values
            The value each slider in self.visualiser_1d.sliders should be set to
        """
        self.visualiser_1d.freeze_input_axes_switch.active = False
        self.visualiser_1d.freeze_output_axes_switch.active = False
        self.visualiser_1d.input_pickers[0].update(value=list(self.input_scalar_menu.items())[selected_scalar_ind][0])
        self.visualiser_1d.axis_resolutions[0].update(value=selected_scalar_res)
        self.visualiser_1d.output_scalar_picker.update(value=self.output_scalar_picker.value)

        for slider, value in zip(self.visualiser_1d.sliders, slider_values):
            slider.value = value

        self.visualiser_1d.update_output()
        self.visualiser_1d.update_output_axes()
        self.visualiser_1d.update_input_axes()

        self.root.active = 1

    def handle_main_figure_click(self, event) -> None:
        """
        Handles a click of in the main figure. Determines whether the click was on the main plot, in which case the
        cross-section line profile plots are updated, or whether an axis was clicked, in which case the view is switched
        to the 1D tab

        Parameters
        ----------
        event
        """
        if event is None:
            self.x_span.update(visible=False)
            self.y_span.update(visible=False)
            return

        if (
            event.x > self.main_figure.x_range.end or
            event.y > self.main_figure.y_range.end or
            (event.x < self.main_figure.x_range.start and event.y < self.main_figure.y_range.start)
        ):
            return

        if event.x < self.main_figure.x_range.start:
            values = [slider.value for slider in self.sliders]
            values[self.input_scalar_ind2] = event.y
            self.configure_1d_panel(self.input_scalar_ind1, self.axis_resolutions[0].value, values)
            return
        if event.y < self.main_figure.y_range.start:
            values = [slider.value for slider in self.sliders]
            values[self.input_scalar_ind1] = event.x
            self.configure_1d_panel(self.input_scalar_ind2, self.axis_resolutions[1].value, values)
            return

        x_ind = self.nearest_bin_index(self.xbins, event.x)
        y_ind = self.nearest_bin_index(self.ybins, event.y)

        self.x_span.update(location=self.xbins[x_ind], visible=True)
        self.y_span.update(location=self.ybins[y_ind], visible=True)
        self.update_line_profiles()

    def update_line_profiles(self) -> None:
        """
        Updates the line-profile cross-sections data based on the location of the cross-hair
        """
        x, y = self.x_span.location, self.y_span.location

        self.y_line_profile.data_source.data = {
            'y': self.ybins,
            'x': self.last_output[self.nearest_bin_index(self.xbins, x), :]
        }
        self.x_line_profile.data_source.data = {
            'x': self.xbins,
            'y': self.last_output[:, self.nearest_bin_index(self.ybins, y)]
        }
