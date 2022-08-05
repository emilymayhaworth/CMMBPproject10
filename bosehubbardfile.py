"""Implementing the 1D Bose-Hubbard model"""


import numpy as np

class BoseHubbardModel: 
    """
    The Hamiltonian reads 

    H = - t \\sum_{i}(\\creation_i \\annihilation_{i+1} + \\annihilation_i \\creation_{i+1} +U/2 \\sum_{i} n_i(n_i - 1) - \\mu \\sum_{i} n_i

    Parameters 
    ----------
    L : int 
        Number of sites
    t, U, mu : float
        Coupling parameters of the above defined Hamiltonian
    nc : int
        Maximum number of bosons per site  

    Attributes 
    ----------
    L : int 
        Number of sites
    d : int
        Local (physical) dimension (= (nc +1) for spinless bosons with maximum occupation number n_c (+1 for no occupation))
    creation, annihilation, n, id :
        Local operators, namely the bosonic creation, annihilation and number operators, and the identity
    H_bonds : list of np.Array[ndim=4]
        The Hamiltonian written in terms of local 2-site operators, ''H = sum_i H_bonds[i]''.
        Each ''H_bonds[i]'' has (physical) legs (i out, (i+1) out, i in, (i+1) in), 
        in short ''i j i* j*''.

    
    
    
    #TODOcheck (esp. last)
    
    
    
    
    """

    def __init__(self, L, t, U, mu, nc):
        self.nc = nc
        d = nc + 1
        self.L = L
        self.d = d
        self.id = np.eye(d)
        self.t, self.U, self.mu = t, U, mu
        self.init_H_bonds()
        self.init_H_mpo()
        self.init_annihilation_op()
        self.init_creation_op()
        self.init_number_op()
        self.init_fixed_H_mpo()

    def init_creation_op(self): 
        #creation operator on site i - written as matrix 
        v = []
        #print(self.nc)
        #print(type(self.nc))
        for i in range(self.nc):
            num = np.sqrt(i+1)
            v.append(num)
        return np.diag(v, -1)

    def init_annihilation_op(self): 
        #annihilation operator on site i - written as matrix 
        v = []
        for i in range(self.nc):
            num = np.sqrt(i+1)
            v.append(num)
        return np.diag(v, 1)

    def init_number_op(self): 
        #number operator on site i - written as matrix 
        v = []
        for i in range(self.nc + 1):
            num = i
            v.append(num)
        return np.diag(v, 0)   

    
    def init_H_bonds(self):
        cr = self.init_creation_op() 
        ann = self.init_annihilation_op()
        num = self.init_number_op()
        d = self.d
        id = self.id
        H_list = []
        for i in range(self.L - 1):
            muL = muR = 0.5 * self.mu
            UL = UR = 0.5 * self.U
            if i == 0: # first bond
                muL = self.mu
                UL = self.U
            if i + 1 == self.L - 1: # last bond
                muR = self.mu
                UR = self.U
            n_squared = num @ num
            #n_minusone = num - id
            H_bond = -self.t * (np.kron(cr, ann) + np.kron(ann, cr)) + (UL/2) * (np.kron(n_squared, id) - np.kron(num, id)) - muL * np.kron(num, id) + (UR/2) * (np.kron(id, n_squared) - np.kron(id, num)) - muR * np.kron(id, num) 
            # H_bond has legs ``i, j, i*, j*``
            H_list.append(np.reshape(H_bond, [d, d, d, d]))
            #print(H_list)
        self.H_bonds = H_list
    
    def energy(self, psi):
        """Evaluate energy E = <psi|H|psi> for the given MPS."""
        assert psi.L == self.L
        #uses function from a mps 
        return np.sum(psi.bond_expectation_value(self.H_bonds)) #psi must = my_mps 

    def new_energy(self, psi): 
        assert psi.L == self.L
        conjugate = np.conj.psi
        
    
    def init_H_mpo(self):
        """Initialize `H_mpo` Hamiltonian. Called by __init__()."""
        #inspired by sol9_dmrg
        w_list = []
        #nc = self.nc
        w = np.zeros((4, 4, self.d, self.d), dtype="float")
        w[0, 0,:, :] = w[3, 3, :, :] = np.eye(self.d, self.d)
        w[0, 1, :, :] = -self.t * self.init_creation_op()
        w[0, 2, :, :] = -self.t * self.init_annihilation_op()
        w[1, 3, :, :] = self.init_annihilation_op()
        w[2, 3, :, :] = self.init_creation_op()
        w[0, 3, :, :] = (self.U/2) * self.init_number_op()**2 - (((self.U/2) + self.mu) * self.init_number_op())
        for site in range(self.L):
            tensor = w
                          
            #t only on one type of op each so hermitian and not squared

            w_list.append(tensor)
        self.H_mpo = w_list     


    def init_fixed_H_mpo(self):
        """Initialize `H_mpo` Hamiltonian. Called by __init__()."""
        #inspired by sol9_dmrg
        w_list = []
        #nc = self.nc
        w = np.zeros((4, 4, self.d, self.d), dtype="float")
        w[0, 0,:, :] = w[3, 3, :, :] = np.eye(self.d, self.d)
        w[0, 1, :, :] = -self.t * self.init_creation_op()
        w[0, 2, :, :] = -self.t * self.init_annihilation_op()
        w[1, 3, :, :] = self.init_annihilation_op()
        w[2, 3, :, :] = self.init_creation_op()
        w[0, 3, :, :] = (self.U/2) * self.init_number_op()**2 - (((self.U/2) + self.mu) * self.init_number_op())
        for site in range(self.L):
            if site != 0 and site != self.L-1:
                tensor = w
            elif site == 0: 
                tensor = w[0, :, :, :]
                tensor = np.expand_dims(tensor, axis = 0)
            elif site == self.L-1:
                tensor = w[:, 3, :, :]   
                tensor = np.expand_dims(tensor, axis = 1)
             
            #t only on one type of op each so hermitian and not squared

            w_list.append(tensor)
        self.fixed_H_mpo = w_list   
#nc = 5
#BoseHubbardModel.init_H_bonds(15)
    

#print(type(nc))



#print(BoseHubbardModel.init_number_op(nc))


""" # local two-site and single-site terms
    lopchains = [OpChain([-t*b_dag, b_ann], [ 1]),
                 OpChain([b_ann, -t*b_dag], [-1]),
                 OpChain([0.5*U*np.dot(numop, numop - np.identity(d)) - mu*numop], [])]
    # convert to MPO
    return local_opchains_to_MPO(qd, L, lopchains)"""


    #question = whz in b model are the first and last bonds g 
    #and the middle all g/2?
    #is it because they add up in the middle to g ? 
    #so which for BH would need to be half? because in ising J isnt 
    #only number operators?
    #
    #do i even need it anymore?! if have MPO part?


