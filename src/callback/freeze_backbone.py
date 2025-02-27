from callback.callback_lib import Callback

class FreezeBackbone(Callback):
    def on_adaptation_start(self, module):
        """
        Freeze the backbone of the network and
        configure optimizers to only update the head parameters
        """
        module.net.freeze_backbone()
        module.net.trainability_info()