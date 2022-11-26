import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
import chex
from flax import struct

"""
BYTECODE
"""
bytecode = "6005600401"


"""
EXAMPLE STACK
"""
stack=jnp.zeros((1024,32),dtype=jnp.uint8)
stack = stack.at[0, 31].set(71)
stack = stack.at[1, 31].set(45)
stack = stack.at[2, 30].set(69)
stack = stack.at[0, 30].set(71)


"""
Useful functions
"""
# :: bytecode -> [uint8 of opcode/operands]
def convert_bytecode_to_jax_array(bytecode: str) -> chex.Array:
    bytecode_byte_list = [int(f"0x{bytecode[i]}{bytecode[i+1]}",0) for i in range(0, len(bytecode), 2)]
    numpy_bytecode = np.array(bytecode_byte_list).astype(dtype=np.uint8)
    jax_bytecode = jnp.array(numpy_bytecode, dtype=jnp.uint8)
    return jax_bytecode

# def see_stack(stack: chex.Array):


def uint32_to_array(n: int) -> chex.Array:
    dividers = [16777216, 65536, 256, 1]
    y = jnp.zeros(32, dtype=jnp.uint8)
    for i,div in enumerate(dividers):
        y = y.at[28+i].set(n//div)
        n = n % div
    return y

"""
Example state
"""
@struct.dataclass
class State:
    stack_height: int = 3
    pc: int = 0


push_amount: int = 2
print(18247 // 65536)
print(uint32_to_array(18247))
f = lambda *x,y: x[0]+x[1]+y

print(f(*[2,4,6,8,800],y=900))
x1 = "00000000000000000000000000000000000000000000000000000000000000004200000000000000000000000000000000000000000000000000000000000000"
x2 = "00000000000000000000000000000000000000000000000000000000000000000042000000000000000000000000000000000000000000000000000000000000"
x3 = "00000000000000000000000000000000000000000000000000000000000000000000420000000000000000000000000000000000000000000000000000000000"
xk = "0000000000000000000000000000000000000000000000000000000000000000"
print(len(xk))


