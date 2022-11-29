class example_dataset():
    def __init__(self, pars):
		#initialize dataset with standard values for parameters
		sc_dataset.__init__(self, pars)
        self.dlabel = 'example' #dataset identifier
        
		#outside inputs
        if self.noise == -1: self.noise = 1
        if self.Ntrain == -1: self.Ntrain = 20        
        if self.dropout_mode == -1: self.dropout_mode = 1
        if self.f_dropout == -1: self.f_dropout = 0
		
        #general information about dataset
		self.Ntest = ...	#number of testpoints
        self.MT_dim = ...     #number of untransformed outputs
        self.MT_dim_trans = ...      #number of transformed outputs (including auxiliary variables)
        self.constant_C = 1
		
		#initialize dataset specific parameter for curves and constraint		
        self.init_pars()
		
        #define input space
		self.xmin = ...		#min x value
		self.xmax = ...		#max x value
		self.xmin_off = ...		#min x offset for test data
		self.xmax_off = ...		#max x offset for test data
		
		#create input space and train + test data (with noise)
		self.train_and_test()
		
		#introduce dropout and generate transformed data
		self.dropout_and_trans()
        
		
		#meta data
		self.unconstrained_as_auxvar = ...		#y/n: use outputs as learned by unconstrained GP as auxiliary outputs
		self.ilist_aux = ...		#list of indices of auxvar dims in y
		self.ilist_trans_aux = ...		#list of indices of auxvar dims in y'
		
		self.virtual_measurements = ...	#y/n: whether virtual measurements should be created
		self.ilist_vm = ...		#list of indices of outputs from which virtual measurements should be created
		self.ilist_vm_trans = ...			#list of indices of transformed outputs where virtual measurements should be inserted
		
		self.vm_crit = ...		#value that auxiliary output has to cross in order for virtual measurement to be created in transformed output
		self.vm_val = ...		#value to be inserted for virtual measurement
		
		self.drop_aux = ...		#y/n: whether some auxiliary outptus should be omitted from y'
		self.ilist_drop_aux = ...   #list of indices of dims in y corresponding to dims in y' that should be dropped
		self.ilist_drop_aux_trans = ...		#list of indices of dims in y' that should be omitted during learning
		
        self.Laplace_opt_vec = [0,1,0,1] if self.drop_aux == 0 else [1,1]   #define what transformation applied to inputs for Laplace approximation
        
	#functions defining dataset and constraint
	def init_pars(self):
		#initilize parameters required for calculating outputs from inputs
		...
		
		#define constraint (if constant)
		self.C = ...
		self.F = ...	
	
	def fC(self, x): #(optional, only in case of non-constant C)
		#function to calculate constraint C(x)
		...
		return C
	
	def f(self, x):
		#calculate outputs F(x)
		return F
		
	def ftrans(self, F):
		#calculate transformed outputs Ftrans(F)
		return Ftrans
		
	def fback(self, Ftrans):
		#calculated backtransformed outputs F(Ftrans)
		return F
	
	def drop_ind_trans(self, drop_ind):
		#calculate dropout indices for transformed outputs: drop_ind_trans(drop_ind)
		return drop_ind_trans
		
	#functions that can be defined optionally, to be used instead of defaults
	def get_optimizer_pars(self, jkernel):
		lr  = ... 	#learning rate
		s1 = ...	#iterations after which scheduler takes step
		s2 = ...	#scheduler factor
		return lr, s1, s2