# importing Qiskit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from math import pi,log,ceil,log2,sqrt
import matplotlib as mpl
from matplotlib import cm, colorbar, colors
# import basic plot tools
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
import sys

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
        len_side = round(num_states**(1./3))
        shape = (len_side, len_side, len_side)

    states = qwalk(dim, power, len_side, iterations)

    all_inds = range(dim+power)
    pos_inds = all_inds[dim+power-1:dim-1:-1]
    counter = 0

    for state in states:
        data = np.around(np.array(state.probabilities(pos_inds)), 5)
        data = np.reshape(data, shape)
        data_list = data.tolist()
        
        if(dim==1 or dim==2):
            plt.title("Step "+str(counter+1))
            pixel_plot = plt.imshow(data, cmap='hot')
            cb = plt.colorbar(pixel_plot)
            plt.savefig('./images/dist'+str(counter)+'.png')
            cb.remove()
        else:
            fig = plt.figure()
            ax = fig.add_axes([0.1,0.1,0.7,0.7], projection='3d')
            ax.set_aspect('equal')
            z,x,y = np.nonzero(data)
            norm=plt.Normalize(0,1)
            cmap = colors.LinearSegmentedColormap.from_list("", ["yellow","red"])
            plotCubes(ax, x,y,z, data=data, cmap=cmap)
            ax.set_ylim(bottom=0, top=shape[0])
            ax.set_xlim(left=0, right=shape[0])
            ax.set_zlim(zmin=0,zmax=shape[0])
            ax.set_title("Step "+str(counter+1))
            ax_cb = fig.add_axes([0.8, 0.3, 0.05, 0.45])

            
            cbar = colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm,orientation='vertical')  
            cbar.set_ticks(np.concatenate((np.unique(data), np.array([1]))), axis=None)
            # set the colorbar transparent as well
            cbar.solids.set(alpha=1)  
            plt.savefig('./images/dist'+str(counter)+'.png')
        counter +=1

def cuboid_data(center, size=(1,1,1)):
    # made by https://stackoverflow.com/questions/30715083/python-plotting-a-wireframe-3d-cuboid?noredirect=1&lq=1
    """
    Create a data array for cuboid plotting.
    ============= ================================================
    Argument      Description
    ============= ================================================
    center        center of the cuboid, triple
    size          size of the cuboid, triple, (x_length,y_width,z_height)
    :type size: tuple, numpy.array, list
    :param size: size of the cuboid, triple, (x_length,y_width,z_height)
    :type center: tuple, numpy.array, list
    :param center: center of the cuboid, triple, (x,y,z)
    """
    # get the (left, outside, bottom) point
    o = list(center)
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
    z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
    return x, y, z


def plotCubeAt(pos=(0,0,0), c="b", alpha=0.1, ax=None):
    # taken from https://stackoverflow.com/questions/40853556/3d-discrete-heatmap-in-matplotlib
    # Plotting N cube elements at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( (pos[0],pos[1],pos[2]) )
        ax.plot_surface(np.array(X), np.array(Y), np.array(Z), color=c, rstride=1, cstride=1, alpha=alpha)

def plotCubes(ax, x,y,z, data, cmap):
    norm = colors.Normalize(0, 1)
    color_fun = lambda i,j,k : cm.ScalarMappable(norm=norm,cmap = cmap).to_rgba(data[i,j,k])
    for xi,yi,zi in zip(x,y,z):
        plotCubeAt(pos=(xi,yi,zi), c=color_fun(xi,yi,zi), alpha=0.9, ax=ax)