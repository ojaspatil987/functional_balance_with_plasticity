{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ojaspatil987/functional_balance_with_plasticity/blob/master/network.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from brian2 import *\n",
        "from \"/content/params.py\" import *\n",
        "# Paper's LIF parameters (Table 1)\n",
        "tau = 20*ms\n",
        "u_th = 20*mV\n",
        "u_reset = 0*mV\n",
        "t_ref = 0*ms\n",
        "eqs = '''\n",
        "du/dt = -u/tau : volt\n",
        "'''\n",
        "neuron = NeuronGroup(1,eqs,threshold='u>uth',reset=\"u=ureset\",refractory=t_ref,method=\"exact\")\n",
        "neurons = NeuronGroup(n,eqs,threshold='u>uth',reset=\"u=ureset\",refractory=t_ref,method=\"exact\")\n",
        "neurons.u= 0*mV\n",
        "syn_model = 'w: volt'\n",
        "on_pre_eq = 'u_post +=w'\n",
        "\n",
        "# E -> E\n",
        "S_EE = Synapses(exc, exc, model=syn_model, on_pre=on_pre_eq)\n",
        "S_EE.connect(condition='i!=j', eps_ee)\n",
        "S_EE.w = J_exc\n",
        "\n",
        "#E ->I\n",
        "S_EI = Synapses(exc,inh, model=syn_model=on_pre=on_pre_eq)\n",
        "S_EI.connect(condition='i!=j', eps_ei)\n",
        "S_EI.w = J_inh\n",
        "\n",
        "#I ->E\n",
        "S_IE=Synapses(inh, exc, model=syn_model, on_pre=on_pre_eq)\n",
        "S_IE.connect(condition='i!=j', eps_ie)\n",
        "S_IE.w = J_inh\n",
        "\n",
        "#I ->I\n",
        "S_II=Synapses(inh,inh,model=syn_model,on_pre=on_post)\n",
        "S_II.connect(\"conditon=i!=j\",eps_ii)\n",
        "S_II.w="
      ],
      "metadata": {
        "id": "I2MBUmih_F9F"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}