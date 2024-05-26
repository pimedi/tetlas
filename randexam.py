import torch
import numpy as np

#######################################################
#####PyTorch와 NumPy를 사용한 난수 생성 예시 ###########
#######################################################

######PyTorch######

# 0과 1 사이의 실수 난수 생성
random_float = torch.rand(1)
print(random_float)
print(type(float(random_float)))
# 지정한 범위 내에서 정수 난수 생성
random_int = torch.randint(1, 10, (1,2))
print(random_int)
print(type(random_int))
# 정규 분포를 따르는 난수 생성
random_normal = torch.randn(1)
print(random_normal)
print(type(random_normal))
# 다차원 텐서의 난수 생성
random_tensor = torch.rand(3, 3)
print(random_tensor)
print(type(random_tensor))


######NumPy######

# 0과 1 사이의 실수 난수 생성
random_float = np.random.rand()
print(random_float)

# 지정한 범위 내에서 정수 난수 생성
random_int = np.random.randint(1, 10)
print(random_int)

# 지정한 범위 내에서 실수 난수 생성
random_uniform = np.random.uniform(1.0, 10.0)
print(random_uniform)

# 정규 분포를 따르는 난수 생성
random_normal = np.random.randn()
print(random_normal)

# 다차원 배열의 난수 생성
random_array = np.random.rand(3, 3)
print(random_array)
