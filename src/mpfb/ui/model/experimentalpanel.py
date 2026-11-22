"""Separate sidebar home for MPFB Character Studio prototypes.

The regular Model panel keeps its upstream role; experiments are intentionally
opt-in and isolated in their own Blender N-panel tab.
"""

import bpy

from ... import ClassManager
from ..abstractpanel import Abstract_Panel


EXPERIMENTAL_CATEGORY = "Experimental Features"
EXPERIMENTAL_PANEL = "MPFB_PT_Experimental_Features_Panel"


class MPFB_PT_Experimental_Features_Panel(Abstract_Panel):
    """Entry point for Character Studio's experimental modelling tools."""

    bl_idname = EXPERIMENTAL_PANEL
    bl_label = "Experimental Features"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = EXPERIMENTAL_CATEGORY
    bl_order = 20

    def draw(self, context):
        layout = self.layout
        layout.label(text="Character Studio research tools", icon='MODIFIER')
        layout.label(text="Non-destructive workflows; keep backups")
        layout.label(text="Select an MPFB human to edit")


ClassManager.add_class(MPFB_PT_Experimental_Features_Panel)