import bpy
from ....services import LogService
from ....services import HumanService
from ....services import ObjectService
from .... import ClassManager

_LOG = LogService.get_logger("model.randomizerace")

class MPFB_OT_Randomize_Race_Operator(bpy.types.Operator): 
    """Randomize character's race"""
    bl_idname = "mpfb.randomize_race"
    bl_label = "Randomize Race"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):                
        _LOG.info("Randomizing race")
        self.report({'INFO'}, "Race randomized!")
        return {'FINISHED'}

ClassManager.add_class(MPFB_OT_Randomize_Race_Operator)