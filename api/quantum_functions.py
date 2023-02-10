# importing Qiskit
import matplotlib.pyplot as plt
import numpy as np
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from math import pi,log,ceil,log2,sqrt
import matplotlib as mpl
from matplotlib import cm
# import basic plot tools
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector

mpl.use('Agg')

# QUANTUM HELPERS
def increment(n_adder):
    qc = QuantumCircuit(n_adder)
    if(n_adder>2):
        for i in range(n_adder-2):
            qc.mcx(list(range(i+1, n_adder)),i)
    if(n_adder>1):
        qc.cx(n_adder-1, n_adder-2)
    qc.x(n_adder-1)

    U_inc = qc.to_gate()
    U_inc.name = "U$_{inc}$"
    return U_inc

def decrement(n_sub):
    qc = QuantumCircuit(n_sub)
    qc.x(n_sub-1)
    if(n_sub>1):
        qc.cx(n_sub-1, n_sub-2)
    if(n_sub>2):
        for i in range(n_sub-3, -1, -1):
            qc.mcx(list(range(i+1, n_sub)),i)
               
    U_dec = qc.to_gate()
    U_dec.name = "U$_{dec}$"
    return U_dec

def shift1D(n_pos):
    n_dir = 1

    qr = QuantumRegister(n_dir+n_pos)
    q_dir = qr[:n_dir]
    q_pos = qr[n_dir:]
    qc = QuantumCircuit(qr)

    # if direction is 0 (LEFT)
    qc.x(q_dir)
    qc.append(decrement(n_pos).control(n_dir), q_dir+q_pos)
    qc.x(q_dir)
    # if direction is 1 (RIGHT)
    qc.append(increment(n_pos).control(n_dir), q_dir+q_pos)

    U_shift = qc.to_gate()
    U_shift.name = "U$_{shift}$"
    return U_shift

def shift2D(n_pos, len_side):
    n_dir = 2 
    HALF_N = ceil(n_pos/2)

    qr = QuantumRegister(n_dir+n_pos)
    q_dir = qr[:n_dir]
    q_pos = qr[n_dir:]
    qc = QuantumCircuit(qr)
    
    # if direction is 00 (RIGHT)
    qc.x(q_dir)
    qc.append(increment(HALF_N).control(2), q_dir+q_pos[HALF_N:])
    qc.x(q_dir)
    # if direction is 01 (DOWN)
    qc.x(q_dir[0])
    qc.append(increment(HALF_N).control(2), q_dir+q_pos[:-HALF_N])
    qc.x(q_dir[0])
    # if direction is 10 (LEFT)
    qc.x(q_dir[1])
    qc.append(decrement(HALF_N).control(2), q_dir+q_pos[HALF_N:])
    qc.x(q_dir[1])
    # if direction is 11 (UP)
    qc.append(decrement(HALF_N).control(2), q_dir+q_pos[:-HALF_N])
    
    
    U_shift = qc.to_gate()
    U_shift.name = "U$_{shift}$"
    return U_shift

def shift3D(n_pos,len_side):
    n_dir = 3
    THIRD_LEN = int(n_pos/3)

    # if 2x2x2 cube, then 8 positions, each side is length 2, so only need 1 qubit to represent each side
    qr = QuantumRegister(n_dir+n_pos)
    q_dir = qr[:n_dir]
    q_pos = qr[n_dir:]
    RL_DIM = q_pos[:THIRD_LEN]
    UD_DIM = q_pos[THIRD_LEN:THIRD_LEN*2]
    IO_DIM = q_pos[-THIRD_LEN:]
    
    qc = QuantumCircuit(qr)
    
    # if direction is 000 (RIGHT)
    qc.x(q_dir)
    qc.append(increment(THIRD_LEN).control(3), q_dir+RL_DIM)
    qc.x(q_dir)
    
    # if direction is 001 (DOWN)
    qc.x(q_dir[0])
    qc.x(q_dir[1])
    qc.append(decrement(THIRD_LEN).control(3), q_dir+UD_DIM)
    qc.x(q_dir[0])
    qc.x(q_dir[1])
    
    # if direction is 010 (LEFT)
    qc.x(q_dir[0])
    qc.x(q_dir[2])
    qc.append(decrement(THIRD_LEN).control(3), q_dir+RL_DIM)
    qc.x(q_dir[0])
    qc.x(q_dir[2])
    
    # if direction is 011 (UP)
    qc.x(q_dir[0])
    qc.append(increment(THIRD_LEN).control(3), q_dir+UD_DIM)
    qc.x(q_dir[0])
    
    # if direction is 100 (BACK)
    qc.x(q_dir[1])
    qc.x(q_dir[2])
    qc.append(decrement(THIRD_LEN).control(3), q_dir+IO_DIM)
    qc.x(q_dir[1])
    qc.x(q_dir[2])
    
    # if direction is 101 (FORWARD)
    qc.x(q_dir[1])
    qc.append(increment(THIRD_LEN).control(3), q_dir+IO_DIM)
    qc.x(q_dir[1])
    
    # if direction is 110
    
    # if direction is 111
    
    
    U_shift = qc.to_gate()
    U_shift.name = "U$_{shift}$"
    return U_shift

def round_remove_zeroes(np_dict):
    for _,d in np.ndenumerate(np_dict):
        for k,v in d.items():
            d[k] = round(v, 5)
    
    new_dict = {}
    for _,d in np.ndenumerate(np_dict):
        for k,v in d.items():
            if v:
                new_dict[k] = v
                
    return new_dict

def qwalk(dim, power, len_side, iterations):
    n_dir = dim
    n_pos = power
    qwalk_reg = QuantumRegister(n_dir+n_pos+1)

    # allocating qubits
    q_dir = qwalk_reg[:n_dir]
    q_pos = qwalk_reg[n_dir:n_dir+n_pos]
    q_anc = qwalk_reg[n_dir+n_pos]

    # lists of indices
    dir_ind = list(range(n_dir))
    pos_ind = list(range(n_dir+n_pos))[n_dir:]
    anc_ind = n_dir+n_pos

    qwalk_circ = QuantumCircuit(qwalk_reg, ClassicalRegister(n_pos))

    # set up the initial position
        
    # set up the initial direction(s)
    for qubit in q_dir:
        qwalk_circ.h(qubit)

    states = []

    states.append(Statevector.from_instruction(qwalk_circ))
    for i in range(iterations):
        
        ''' Uncomment to add target state (Sticky walk) functionality'''
        #qwalk_circ.mcx(pos_ind, anc_ind) oracle, target is |11111..1>
        
        '''controlled shift where ancilla is control
        only shift if we are in a non-target state'''
        qwalk_circ.x(q_anc)

        if (dim == 1): # if 1D walk, use 1D shift
            qwalk_circ.append(shift1D(n_pos).control(1), [anc_ind]+dir_ind+pos_ind)
        elif (dim == 2): # if 2D walk, use 2D shift
            qwalk_circ.append(shift2D(n_pos,len_side).control(1), [anc_ind]+dir_ind+pos_ind)
        else:
            qwalk_circ.append(shift3D(n_pos,len_side).control(1), [anc_ind]+dir_ind+pos_ind)

        # add state to list
        states.append(Statevector.from_instruction(qwalk_circ))
        
        qwalk_circ.x(q_anc)
        
        '''Uncomment to add target state (Sticky walk) functionality'''
        #qwalk_circ.reset(q_anc)
        qwalk_circ.h(q_dir)

    return states

def create_plots(dim, num_states, iterations):
    power = int(log2(num_states))
    len_side = 0
    shape = ()
    if (dim == 1):
        len_side = num_states
        shape = (1, len_side)
    elif (dim == 2):
        len_side = int(sqrt(num_states))
        shape = (len_side,len_side)
    else:
        len_side = int(num_states**(1./3))
        shape = (len_side, len_side, len_side)

    states = qwalk(dim, power, len_side, iterations)

    all_inds = range(dim+power)
    pos_inds = all_inds[dim+power-1:dim-1:-1]
    counter = 0

    for state in states:
        data = np.around(np.array(state.probabilities(pos_inds)), 5)
        data = np.reshape(data, shape)
        
        if(dim==1 or dim==2):
            plt.title("Current States")
            pixel_plot = plt.imshow(data, cmap='hot')
            cb = plt.colorbar(pixel_plot)
            plt.savefig('./images/dist'+str(counter)+'.png')
            cb.remove()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            z,x,y = data.nonzero()
            ax.scatter(x,y,z, zdir='z', c = 'green', marker='s', s=500)
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0)
            ax.set_zlim(zmin=0)
            #plt.colorbar()
            ax.set_title("3D Heatmap")
            plt.savefig('./images/dist'+str(counter)+'.png')
        counter +=1

def create_plots2D(states):
    # PLOTTING FOR 8 x 8 GRID
    sum_data = np.zeros((8,8))
    meta_states = []
    counter = 0

    # states = states[len(states)-1:]
    for state in states:
        
        np_dict = round_remove_zeroes(np.array(state.probabilities_dict([7,6,5,4,3,2])))         
        #print("Position qubits:",np_dict)
        
        data = np.around(np.array(state.probabilities([7,6,5,4,3,2])), 5)
        data = np.reshape(data, (8,8))
        sum_data = sum_data + data
        meta_states.append(sum_data)
        d = state.probabilities_dict([7,6,5,4,3,2])
        
        # customizing plot
        plt.title("Current States")
        pixel_plot = plt.imshow(data, cmap='hot')
        cb = plt.colorbar(pixel_plot)
        plt.savefig('./images/dist'+str(counter)+'.png')
        cb.remove()
        counter +=1