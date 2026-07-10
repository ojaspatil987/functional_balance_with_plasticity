from brian2 import *
from params.py import *
# Paper's LIF parameters (Table 1)
tau = 20*ms
u_th = 20*mV
u_reset = 0*mV
t_ref = 0*ms
eqs = '''
du/dt = -u/tau : volt
'''
neuron = NeuronGroup(1,eqs,threshold='u>uth',reset="u=ureset",refractory=t_ref,method="exact")
neurons = NeuronGroup(n,eqs,threshold='u>uth',reset="u=ureset",refractory=t_ref,method="exact")
neurons.u= 0*mV
syn_model = 'w: volt'
on_pre_eq = 'u_post +=w'

# E -> E
S_EE = Synapses(exc, exc, model=syn_model, on_pre=on_pre_eq)
S_EE.connect(condition='i!=j', eps_ee)
S_EE.w = J_exc

#E ->I
S_EI = Synapses(exc,inh, model=syn_model=on_pre=on_pre_eq)
S_EI.connect(condition='i!=j', eps_ei)
S_EI.w = J_inh

#I ->E
S_IE=Synapses(inh, exc, model=syn_model, on_pre=on_pre_eq)
S_IE.connect(condition='i!=j', eps_ie)
S_IE.w = J_inh

#I ->I
S_II=Synapses(inh,inh,model=syn_model,on_pre=on_post)
S_II.connect("conditon=i!=j",eps_ii)
S_II.w=
