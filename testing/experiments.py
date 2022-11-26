import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
import chex

bytecode = "608060405234801561001057600080fd5b50610209806100206000396000f3fe608060405234801561001057600080fd5b506004361061004c5760003560e01c806306661abd14610051578063371303c01461006f5780636d4ce63c14610079578063b3bcfa8214610097575b600080fd5b6100596100a1565b60405161006691906100f5565b60405180910390f35b6100776100a7565b005b6100816100c2565b60405161008e91906100f5565b60405180910390f35b61009f6100cb565b005b60005481565b60016000808282546100b99190610110565b92505081905550565b60008054905090565b60016000808282546100dd9190610166565b92505081905550565b6100ef8161019a565b82525050565b600060208201905061010a60008301846100e6565b92915050565b600061011b8261019a565b91506101268361019a565b9250827fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff0382111561015b5761015a6101a4565b5b828201905092915050565b60006101718261019a565b915061017c8361019a565b92508282101561018f5761018e6101a4565b5b828203905092915050565b6000819050919050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052601160045260246000fdfea264697066735822122024aab70c051342381d44d8a637bcee0d74e10d21518e6c830a04e0d78c112a2564736f6c63430008070033"

bytecode = "6005600401"
# print(list(bytecode))


# print(int("0x60",))
stack=jnp.zeros((1024,32),dtype=jnp.uint8)

stack = stack.at[0, 31].set(71)
stack = stack.at[1, 31].set(45)
stack = stack.at[2, 30].set(69)
stack = stack.at[0, 30].set(71)


arr2 = jnp.zeros(32, dtype=jnp.uint32).at[31].set(1).at[30].set(256).at[29].set(65536).at[28].set(16777216)

def array_to_uint32(x):
    # print(x)
    y = jnp.multiply(arr2, x)
    return jnp.sum(y)

# print(new_stack)
# result = vmap(array_to_uint32)(stack)
# print(result)
# print(jnp.count_nonzero(result))


# :: bytecode -> [uint8 of opcode/operands]
def convert_bytecode_to_jax_array(bytecode: str) -> chex.Array:
    bytecode_byte_list = [int(f"0x{bytecode[i]}{bytecode[i+1]}",0) for i in range(0, len(bytecode), 2)]
    numpy_bytecode = np.array(bytecode_byte_list).astype(dtype=np.uint8)
    jax_bytecode = jnp.array(numpy_bytecode, dtype=jnp.uint8)
    return jax_bytecode

jax_bytecode = convert_bytecode_to_jax_array(bytecode)


stack_height = 3
push_amount = 2
pc = 0


stack = jax.lax.select(
    True,
    stack.at[stack_height, 32-push_amount:32].set(jax_bytecode[pc+1:pc+push_amount+1]),
    stack,
)

print(stack[0:8,:])

