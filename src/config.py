class Config:
    input_size = (224, 224, 3)
    num_classes = 1000
    depth = [3, 4, 6, 3]  # ResNet-50 
    reduction_ratio = 16   # for SE 
