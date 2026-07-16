from brian2 import *
from archive.params import *

eqs = '''
du/dt = -u/tau : volt
'''
neuron = NeuronGroup(1,eqs,threshold='u>uth',reset="u=ureset",method="exact")
neurons = NeuronGroup(n,eqs,threshold='u>uth',reset="u=ureset",method="exact")
neurons.u= 0*mV

exc = neurons[:ne]
inh = neurons[ne:]
J_exc = J*mV          # 0.5 mV
J_inh = -g*J_exc       # -4 mV

syn_model = 'w: volt'
on_pre_eq = 'u_post +=w'
###############################################

#add the synaptic platicity block


############################################

# E -> E
S_EE = Synapses(exc, exc, model=syn_model, on_pre=on_pre_eq)
S_EE.connect(eps_ee)
S_EE.w = J_exc

#E ->I
S_EI = Synapses(exc,inh, model=syn_mode, on_pre=on_pre_eq)
S_EI.connect(eps_ei)
S_EI.w = J_exc

#I ->E
S_IE=Synapses(inh, exc, model=syn_model, on_pre=on_pre_eq)
S_IE.connect(eps_ie)
S_IE.w = J_inh

#I ->I
S_II=Synapses(inh,inh,model=syn_model,on_pre=on_post)
S_II.connect(eps_ii)
S_II.w=J_inh
