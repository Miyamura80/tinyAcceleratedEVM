import jax
import jax.numpy as jnp
from jax import lax, vmap
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, List
import chex
from flax import struct
import numpy as np


"""
CONSTANTS
"""
ARITHMETIC_OPCODES = {
        "0x00" : "STOP",
        "0x01" : "ADD",
        "0x02" : "MUL",
        "0x03" : "SUB",
        "0x04" : "DIV",
        "0x05" : "SDIV",
        "0x06" : "MOD",
        "0x07" : "SMOD",
        "0x08" : "ADDMOD",
        "0x09" : "MULMOD",
}
LOOKUP_OPCODES = {hex(int("0x60",0)+i): f"PUSH{i+1}" for i in range(32)}

MEMORY_OPCODES = {
    "0x52" : "MSTORE",
    "0x53" : "MSTORE8",
}

EVM_OPCODES = {
    **LOOKUP_OPCODES,
    **ARITHMETIC_OPCODES,
}



"""
ENVIRONMENT
"""

@struct.dataclass
class EnvState:
    memory: chex.Array
    stack: chex.Array
    stack_height: int
    pc: int
    gas: int
    terminal: bool
    # transaction: 
    # message:
    # world_state: chex.Array 

"""
At the start of every round of execution, the current instruction is found by 
taking the pcth byte of code (or 0 if pc >= len(code))
"""

@struct.dataclass
class EnvParams:
    memory_capacity: int = 1024
    default_gas: int = 64
    maximum_gas: int = 256        

class tinyAcceleratedEVM(environment.Environment):
    """
    JAX Compatible version of EVM environment 
    https://ethereum.org/en/whitepaper/#code-execution
    ENVIRONMENT DESCRIPTION - 'tinyAcceleratedEVM'

    - Actions are encoded as follows: ['n','t']
        - n: next time step
        - t: terminate (only exist in debug mode)
    """

    def __init__(self, gas_paid: int, bytecode: str = "6005600401600902600206", debug_mode: bool = False):
        super().__init__()
        self.obs_shape = (1024, 1024, 1)
        self.bytecode = bytecode
        self.opcode_list = convert_bytecode_to_opcode_list(bytecode)
        self.jax_bytecode = convert_bytecode_to_jax_array(bytecode)


        # Full action set: ['n','t']
        self.full_action_set = jnp.array([0, 1])
        # Minimal action set: ['n']
        self.minimal_action_set = jnp.array([0])
        
        # Set active action set for environment
        # If minimal map to integer in full action set
        if not debug_mode:
            self.action_set = self.minimal_action_set
        else:
            self.action_set = self.full_action_set

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""
        # Resolve player action - fire, left, right.
        a = self.action_set[action]
        state = step_machine(a, state, params, self.jax_bytecode,self.opcode_list)

        # TODO: Update Reward here
        reward = 0.0

        # Check game condition & no. steps for termination condition
        terminal = self.is_terminal(state, params)

        # Update other metadata for state here
        state = state.replace(
            terminal=terminal,
        )

        info = {"discount": 1 - terminal}
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            terminal,
            info,
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state """
        state = EnvState(
            memory=jnp.zeros((params.memory_capacity, 8),dtype=jnp.uint8),
            stack=jnp.zeros((1024,32),dtype=jnp.uint8),
            stack_height=0,
            pc=0,
            gas=params.default_gas,
            terminal=False,
        )
        return self.get_obs(state), state

    # REVIEW
    def get_obs(self, state: EnvState) -> chex.Array:
        """Return observation from raw state trafo."""
        return jnp.hstack([state.stack, state.memory])

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        pc_terminated = state.pc >= len(self.bytecode) // 2
        return jnp.logical_or(pc_terminated, state.terminal)

    @property
    def name(self) -> str:
        """Environment name."""
        return "tinyAcceleratedEVM"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, self.obs_shape)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "memory": spaces.Box(0,255, (params.memory_capacity, 32)),
                "stack": spaces.Box(0,255, (1024, 32)),
                "stack_height": spaces.Discrete(1024),
                "gas": spaces.Discrete(params.maximum_gas),
                "pc": spaces.Discrete(len(params.bytecode)),
                "terminal": spaces.Discrete(2),
            }
        )

# Main state transition machine
def step_machine(action: int, state: EnvState, params: EnvParams, jax_bytecode: chex.Array, opcode_list) -> EnvState:
    """Update the EVM stack machine"""

    instruction = opcode_list[state.pc]
    s_height = state.stack_height
    gas, incr,inp_num,out_num,op = operation_lookup(instruction)
    delta_h = out_num - inp_num
    new_sh = s_height + delta_h

    pc = state.pc

    # Dealing with conditions for different types of instructions
    push_cond = (incr > 1)
    operation_cond = jnp.logical_and(op != None, inp_num+out_num > 0) # There is something to operate on
    dup_cond = jnp.logical_and(out_num > inp_num, inp_num > 0) # Something new is created
    swap_cond = jnp.logical_and(out_num==inp_num, inp_num >= 2) # Argument number doesn't change, and there is something to swap

    # Handling PUSH instructions
    push_amount = incr-1
    stack = jax.lax.select(
        push_cond,
        state.stack.at[s_height, 32-push_amount:32].set(jax_bytecode[pc+1:pc+push_amount+1]),
        state.stack,
    )

    # Handling operation_cond
    stack = jax.lax.select(
        operation_cond,
        stack
            .at[new_sh:s_height,:].set(0) # Clear top
            .at[new_sh-1,:].set(apply_operation(stack, op, inp_num, out_num, s_height, operation_cond)),
        stack,
    )

    # Increment program counter
    pc += incr


    # POST-STATE TRANSITION

    # Check for stack underflow
    stack_limit_reached = state.stack_height > 1024
    terminal = jnp.logical_or(state.terminal, stack_limit_reached)
    return state.replace(
        memory=state.memory,
        stack=stack,
        stack_height=new_sh,
        pc=pc,
        gas=state.gas,
        terminal=terminal,
    )

def apply_operation(stack: chex.Array, op, inp_num, out_num, s_height, operation_cond) -> chex.Array:
    stack_reduction = inp_num - out_num
    if not operation_cond:
        return stack[s_height-stack_reduction,:]
    uint32_inputs = [int(array_to_uint32(stack[s_height-1-i,:])) for i in range(stack_reduction+1)]
    return uint32_to_array(op(*uint32_inputs))

    


# Return the (min_gas, pc_increment,inp_num,out_num,op)
def operation_lookup(instruction):

    # INSTRUCTIONS 00 - 09
    arithmetic_instructions = {
        "0x01" : (3,1,2,1, lambda *x: x[0]+x[1]),
        "0x02" : (5,1,2,1, lambda *x: x[0]*x[1]),
        "0x03" : (3,1,2,1, lambda *x: x[0]-x[1]),
        "0x04" : (5,1,2,1, lambda *x: x[0]//x[1]),
        "0x05" : (5,1,2,1, lambda *x: x[0]//x[1]), # TODO: Implement truncated for SDIV
        "0x06" : (5,1,2,1, lambda *x: x[0]%x[1]), 
        "0x07" : (5,1,2,1, lambda *x: x[0]%x[1]* np.sign(x[0]*x[1])),
        "0x08" : (8,1,3,1, lambda *x: (x[0]+x[1])%x[2]),
        "0x09" : (8,1,3,1, lambda *x: (x[0]*x[1])%x[2]),
    }

    # memory_instructions = {
    #     "0x52" : (3,1,2,0, lambda x:),
    #     "0x53" : (3,1,2,0),
    # }

    # INSTRUCTIONS 60 - 7F
    push_instructions = {hex(int("0x60",0)+i):(3,i+2,0,1,None) for i in range(32)}

    lookup = {**arithmetic_instructions, **push_instructions}
    return lookup[instruction]

# :: bytecode -> [hex of opcodes/operands]
def convert_bytecode_to_opcode_list(bytecode: str):
    opcode_list = [f"0x{bytecode[i]}{bytecode[i+1]}" for i in range(0, len(bytecode), 2)]
    return opcode_list

# :: bytecode -> [uint8 of opcode/operands]
def convert_bytecode_to_jax_array(bytecode: str) -> chex.Array:
    bytecode_byte_list = [int(f"0x{bytecode[i]}{bytecode[i+1]}",0) for i in range(0, len(bytecode), 2)]
    numpy_bytecode = np.array(bytecode_byte_list).astype(dtype=jnp.uint8)
    jax_bytecode = jnp.array(numpy_bytecode, dtype=jnp.uint8)
    return jax_bytecode

# Takes an array x, with |x| = 32, and returns the uint32 representation
def array_to_uint32(x: chex.Array) -> int:
    converter = jnp.zeros(32, dtype=jnp.uint32).at[31].set(1).at[30].set(256).at[29].set(65536).at[28].set(16777216)
    y = jnp.multiply(converter, x)
    return jnp.sum(y)

# See populated parts of stack
def see_populated_stack(stack: chex.Array, stack_height: int) -> chex.Array:
    return vmap(array_to_uint32)(stack[0:stack_height,:])


# Turn uint32 integer to an array x, with |x| = 32, x :: List[uint8]
# TODO: Optimize this for JAX
def uint32_to_array(n: int) -> chex.Array:
    dividers = [16777216, 65536, 256, 1]
    y = jnp.zeros(32, dtype=jnp.uint8)
    for i,div in enumerate(dividers):
        y = y.at[28+i].set(n//div)
        n = n % div
    return y



if __name__=="__main__":

    bytecode = "6002600560040160090206"
    opcode_list = convert_bytecode_to_opcode_list(bytecode)
    
    evm_sim = tinyAcceleratedEVM(256, bytecode)
    params = evm_sim.default_params
    key = jax.random.PRNGKey(0)


    observation, state = evm_sim.reset_env(key, params)


    action = 0
    terminal = False
    prev_opcode = opcode_list[state.pc]

    while not terminal:
        observation, state, reward, terminal, info = evm_sim.step_env(key,state,action,params)
        _, pc_increment,_,_,_ = operation_lookup(prev_opcode)
        operand = "".join([x[2:] for x in opcode_list[state.pc-pc_increment+1:state.pc]])
        print("="*20)
        print(f"Program Counter: {state.pc} || Instruction: {EVM_OPCODES[prev_opcode]} {operand}")
        stack_visual = see_populated_stack(state.stack, state.stack_height)
        print(f"Stack (From bottom to top): {stack_visual}[->]")
        prev_opcode = opcode_list[state.pc] if not terminal else "0x00"
        t = input("")


