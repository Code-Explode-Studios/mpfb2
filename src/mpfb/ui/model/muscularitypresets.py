"""One-click physique presets based on MPFB's existing anatomical morph targets.

These are art-direction starting points, not anatomical muscle simulations. Each
preset affects only phenotype gender, overall muscle/weight and the seven built-in
muscle detail regions. Other character settings and individual morphs are preserved.
"""

import os

import bpy
from bpy.props import EnumProperty

from ... import ClassManager
from ...entities.objectproperties import HumanObjectProperties
from ...services import HumanService, LocationService, LogService, ObjectService, TargetService, UiService
from ..abstractpanel import Abstract_Panel
from ..mpfboperator import MpfbOperator
from .modelpanel import MODEL_PROPERTIES

_LOG = LogService.get_logger("model.muscularitypresets")

# These are shape-key pairs defined in data/targets/target.json. The preset changes
# both left and right regions identically, regardless of the current symmetry toggle.
MUSCLE_REGIONS = {
    "shoulders": ("arms", ("l-upperarm-shoulder-muscle", "r-upperarm-shoulder-muscle")),
    "upper_arms": ("arms", ("l-upperarm-muscle", "r-upperarm-muscle")),
    "forearms": ("arms", ("l-lowerarm-muscle", "r-lowerarm-muscle")),
    "chest": ("torso", ("torso-muscle-pectoral",)),
    "back": ("torso", ("torso-muscle-dorsi",)),
    "thighs": ("legs", ("l-upperleg-muscle", "r-upperleg-muscle")),
    "calves": ("legs", ("l-lowerleg-muscle", "r-lowerleg-muscle")),
}

# Value scales are normalized MPFB slider values, not measured muscle mass or body-fat percentages.
# The same muscularity choices apply to either phenotype; gender only changes the
# underlying morph interpolation, not the maximum muscularity allowed.
MUSCULARITY_LEVELS = {
    "defined": {
        "label": "Lean / defined",
        "description": "Lean physique with visible definition and moderate muscle volume",
        "muscle": 0.70, "weight": 0.28,
        "regions": {"shoulders": 0.14, "upper_arms": 0.14, "forearms": 0.10,
                    "chest": 0.12, "back": 0.12, "thighs": 0.16, "calves": 0.12},
    },
    "athletic": {
        "label": "Athletic",
        "description": "Balanced athletic musculature and a lean-to-average build",
        "muscle": 0.82, "weight": 0.39,
        "regions": {"shoulders": 0.32, "upper_arms": 0.30, "forearms": 0.25,
                    "chest": 0.30, "back": 0.32, "thighs": 0.35, "calves": 0.27},
    },
    "bodybuilder": {
        "label": "Bodybuilder",
        "description": "High muscle definition and developed individual muscle groups",
        "muscle": 0.99, "weight": 0.50,
        "regions": {"shoulders": 0.60, "upper_arms": 0.64, "forearms": 0.47,
                    "chest": 0.57, "back": 0.58, "thighs": 0.61, "calves": 0.48},
    },
    "power": {
        "label": "Power / mass",
        "description": "Broad, heavily muscled physique with greater overall body mass",
        "muscle": 0.95, "weight": 0.72,
        "regions": {"shoulders": 0.56, "upper_arms": 0.58, "forearms": 0.51,
                    "chest": 0.55, "back": 0.62, "thighs": 0.65, "calves": 0.52},
    },
}

_GENDERS = {"female": 0.0, "male": 1.0}
_PRESET_ITEMS = tuple(
    (gender + "_" + level, gender.title() + " - " + data["label"], data["description"])
    for gender in _GENDERS for level, data in MUSCULARITY_LEVELS.items()
)


def _get_target_path(section, target_name):
    """Find a shipped detail target in either supported MakeHuman format."""
    root = LocationService.get_mpfb_data("targets")
    for extension in (".target.gz", ".target"):
        path = os.path.join(root, section, target_name + extension)
        if os.path.isfile(path):
            return path
    return None


class MPFB_OT_Apply_Muscularity_Preset(MpfbOperator):
    """Apply a physique preset to an existing unbaked MPFB basemesh"""

    bl_idname = "mpfb.apply_muscularity_preset"
    bl_label = "Apply muscularity preset"
    bl_options = {'REGISTER', 'UNDO'}

    preset: EnumProperty(name="Physique", items=_PRESET_ITEMS)

    def get_logger(self):
        return _LOG

    @classmethod
    def poll(cls, context):
        if not context.active_object or context.mode != "OBJECT":
            return False
        basemesh = ObjectService.find_object_of_type_amongst_nearest_relatives(context.active_object, "Basemesh")
        return bool(basemesh and TargetService.has_any_shapekey(basemesh))

    def hardened_execute(self, context):
        # Validate before mutating the character (important for scripted calls).
        if self.preset not in {item[0] for item in _PRESET_ITEMS}:
            self.report({'ERROR'}, "Unknown muscularity preset")
            return {'CANCELLED'}
        gender, level_name = self.preset.split("_", 1)
        level = MUSCULARITY_LEVELS[level_name]
        basemesh = ObjectService.find_object_of_type_amongst_nearest_relatives(context.active_object, "Basemesh")
        if not basemesh or not TargetService.has_any_shapekey(basemesh):
            self.report({'ERROR'}, "Select an unbaked MPFB human first")
            return {'CANCELLED'}

        # Preflight missing files, so a partial installation does not apply the
        # macro changes and then fail halfway through loading muscle regions.
        to_load = {}
        for region, (section, names) in MUSCLE_REGIONS.items():
            for name in names:
                positive = name + "-incr"
                if not TargetService.has_target(basemesh, positive):
                    path = _get_target_path(section, positive)
                    if path is None:
                        self.report({'ERROR'}, "Missing built-in muscle target: " + positive)
                        return {'CANCELLED'}
                    to_load[positive] = path

        prune = MODEL_PROPERTIES.get_value("prune", entity_reference=context.scene)
        ObjectService.activate_blender_object(basemesh)

        # One macro rebuild instead of three expensive rebuilds (which would
        # happen if we assigned each Scene slider separately).
        for name, value in (("gender", _GENDERS[gender]),
                            ("muscle", level["muscle"]), ("weight", level["weight"])):
            HumanObjectProperties.set_value(name, value, entity_reference=basemesh)
        TargetService.reapply_macro_details(basemesh, remove_zero_weight_targets=prune)

        for region, (_, names) in MUSCLE_REGIONS.items():
            strength = level["regions"][region]
            for name in names:
                positive, negative = name + "-incr", name + "-decr"
                # Remove pre-existing opposing values when swapping presets;
                # leave every unrelated body, face and asymmetry target alone.
                if TargetService.has_target(basemesh, negative):
                    TargetService.set_target_value(basemesh, negative, 0.0, delete_target_on_zero=prune)
                if TargetService.has_target(basemesh, positive):
                    TargetService.set_target_value(basemesh, positive, strength)
                else:
                    TargetService.load_target(basemesh, to_load[positive], weight=strength, name=positive)

        if MODEL_PROPERTIES.get_value("refit", entity_reference=context.scene):
            HumanService.refit(basemesh, operator=self)

        self.report({'INFO'}, "Applied " + gender + " " + level["label"] + " physique")
        return {'FINISHED'}


ClassManager.add_class(MPFB_OT_Apply_Muscularity_Preset)


class MPFB_PT_Muscularity_Presets_Panel(Abstract_Panel):
    """One-click presets for male and female muscle morphology."""

    bl_label = "Muscularity presets"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "MPFB_PT_Experimental_Features_Panel"
    bl_category = "Experimental Features"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return MPFB_OT_Apply_Muscularity_Preset.poll(context)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Physique presets (female / male)")
        for level_name, level in MUSCULARITY_LEVELS.items():
            row = layout.row(align=True)
            for gender in _GENDERS:
                op = row.operator("mpfb.apply_muscularity_preset",
                                  text=gender.title() + ": " + level["label"])
                op.preset = gender + "_" + level_name
        layout.label(text="Keeps age, height and other details unchanged")
        layout.label(text="Refit clothes/rig if Auto refit is disabled")


ClassManager.add_class(MPFB_PT_Muscularity_Presets_Panel)
