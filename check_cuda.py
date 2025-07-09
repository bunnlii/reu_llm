import torch
print(torch.__version__)            # Should be 2.2.0 or higher
print(torch.version.cuda)           # Should be 12.8
print(torch.cuda.is_available())    # Should be True

