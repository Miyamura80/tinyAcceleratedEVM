
# Limitations of tinyAcceleratedEVM

Being a specialized software framework for running on the GPU, tinyAcceleratedEVM has a few limitations


## Stack

### Stack Arithmetic

The stack implementations is only capable of 32 bits integer arithmetic per stack level, instead of 256. 

This is due to the limitations of what numpy can handle in terms of integer datatypes.

Thus, the tinyAcceleratedEVM will ignore arithmetic with any values on the stack after the 64th bit from the LSB.

