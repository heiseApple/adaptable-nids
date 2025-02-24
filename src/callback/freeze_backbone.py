from callback.callback_lib import Callback

class FreezeBackbone(Callback):
    def on_adaptation_start(self, module):
        """
        Freeze the backbone of the network and
        configure optimizers to only update the head parameters
        """
        if getattr(module, 'adaptation_strat', None) == 'freezing':
            module.net.freeze_backbone()
            module.net.trainability_info()
            module.configure_optimizers(params=module.net.head.parameters())
