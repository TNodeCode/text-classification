import torch


class ExportToONNXCommand:
    """
    Export a PyTorch model to ONNX with dynamic axes.

    Requires dummy_input: a tensor matching processor output.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dummy_input: torch.Tensor,
        output_path: str,
        opset_version: int = 13,
    ):
        self.model = model
        self.dummy_input = dummy_input
        self.output_path = output_path
        self.opset_version = opset_version

    def invoke(self):
        self.model.eval()
        torch.onnx.export(
            self.model,
            (self.dummy_input,),
            self.output_path,
            input_names=['pixel_values'],
            output_names=['logits', 'pred_boxes'],
            dynamic_axes={
                'pixel_values': {0: 'batch', 2: 'height', 3: 'width'},
                'logits':       {0: 'batch'},
                'pred_boxes':   {0: 'batch', 1: 'num_queries'},
            },
            opset_version=self.opset_version,
            do_constant_folding=True,
        )
