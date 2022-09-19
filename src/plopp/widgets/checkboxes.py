# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

from html import escape
import ipywidgets as ipw
from typing import Callable


class Checkboxes(ipw.HBox):
    """
    Widget providing a list of checkboxes, along with a button to toggle them all.
    """

    def __init__(self, entries: list, description=""):
        self.checkboxes = {}
        self._lock = False
        self._description = ipw.Label(value=description)

        for key in entries:
            self.checkboxes[key] = ipw.Checkbox(value=True,
                                                description=f"{escape(key)}",
                                                indent=False,
                                                layout={"width": "initial"})

        to_hbox = [
            self._description,
            ipw.HBox(list(self.checkboxes.values()),
                     layout=ipw.Layout(flex_flow='row wrap'))
        ]

        if len(self.checkboxes):
            # Add a master button to control all masks in one go
            self.toggle_all_button = ipw.ToggleButton(value=True,
                                                      description="De-select all",
                                                      disabled=False,
                                                      button_style="",
                                                      layout={"width": "initial"})
            for cbox in self.checkboxes.values():
                ipw.jsdlink((self.toggle_all_button, 'value'), (cbox, 'value'))
            to_hbox.insert(1, self.toggle_all_button)

        super().__init__(to_hbox)

    def watch(self, callback: Callable, **kwargs):
        for chbx in self.checkboxes.values():
            chbx.observe(callback, **kwargs)

    @property
    def value(self) -> dict:
        """
        """
        return {key: chbx.value for key, chbx in self.checkboxes.items()}
