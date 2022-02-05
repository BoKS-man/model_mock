import torch
from model_mock import ModelMock
from utils import check_model_size

model = ModelMock(7, 1)
print(round(check_model_size(model), 3), 'Mb')

dummy_tensor = torch.ones(1,32).float()
print(dummy_tensor.mean())

result = model(dummy_tensor)
print(result.mean())