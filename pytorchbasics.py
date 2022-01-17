import torch
import numpy

#CUDA SPECIFIC BASIC COMMANDS___________

# print(torch.cuda.current_device())
# print(torch.cuda.device_count())

# print(torch.cuda.get_device_name())
# print(torch.cuda.is_available())

#NUMPY AND TENSOR RELATIONS______________

""" 
x = torch.rand(5,5).cuda() #.cuda is needed to enfore storing the tensor on the GPU. \
by default both numpy array and pytorch tensor are stgored on the cpu and share the same memory location. 
enforcing the tensor to stay on the gpu makes sure changing one does not change the other
"""
# #print(x.is_cuda)
# y = x.cpu().numpy()
# #print(x)
# #print(y)

# x.add_(1)
# #print(x)
# #print(y)

#b=torch.from_numpy("numpy array").cuda() 

#GRAD AND BACKPROPAGATION____________________

b = torch.randn(5, requires_grad=True).cuda()
b = torch.tensor(0.0, dtype = torch.float,requires_grad=True).cuda()

#print(b)
y = b+2
#print(z)

#v = torch.tensor([0.1,0.2,0.3,0.02,1.2], dtype = torch.float).cuda()
b.retain_grad()
y.backward()
print(y)
print(b)
print(b.grad)

#to get rid of requires grad attribute
# x = torch.randn(5).cuda()
# x.requires_grad_(False)
# print(x)



