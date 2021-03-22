import brainpy as bp
import numpy as np
import bpmodels
import time
import csv

def test_lif(num, device):

    print('Scale:{}, Model:LIF, Device:{}, '.format(num, device) , end = '')
    st_build = time.time()

    bp.profile.set(jit=True,
                device=device,
                dt=0.1,
                numerical_method='exponential')
                
    V_rest = -52.0
    V_reset = -60.0
    V_th = -50.0

    neu = bpmodels.neurons.get_LIF(V_rest=V_rest, V_reset = V_reset, V_th=V_th, noise=0., mode='scalar')
    syn = bpmodels.synapses.get_exponential(tau_decay = 2.0, mode='scalar')

    num_exc = int(num / 2)
    num_inh = int(num / 2)
    prob = 0.01

    JE = 1 / np.sqrt(prob * num_exc)
    JI = 1 / np.sqrt(prob * num_inh)

    group = bp.NeuGroup(neu, geometry=num_exc + num_inh, monitors=['spike'])

    group.ST['V'] = np.random.random(num_exc + num_inh) * (V_th - V_rest) + V_rest

    exc_conn = bp.SynConn(syn,
                        pre_group=group[:num_exc],
                        post_group=group,
                        conn=bp.connect.FixedProb(prob=prob))
    exc_conn.ST['g'] = JE

    inh_conn = bp.SynConn(syn,
                        pre_group=group[num_exc:],
                        post_group=group,
                        conn=bp.connect.FixedProb(prob=prob))
    exc_conn.ST['g'] = -JI

    net = bp.Network(group, exc_conn, inh_conn)

    ed_build = time.time()

    st_run = time.time()
    net.run(duration=1000., inputs=(group, 'ST.input', 3.))
    ed_run = time.time()

    build_time = float(ed_build - st_build)
    run_time = float(ed_run - st_run)

    print('BuildT:{:.2f}s, RunT:{:.2f}s'.format(build_time, run_time))
    return run_time, build_time
    
def test_hh(num, device):

    print('Scale:{}, Model:HH, Device:{}, '.format(num, device) , end = '')
    st_build = time.time()

    bp.profile.set(jit=True,
                device=device,
                dt=0.1,
                numerical_method='exponential')
                
    num_exc = int(num * 0.8)
    num_inh = int(num * 0.2)
    num = num_exc + num_inh
    Cm = 200  # Membrane Capacitance [pF]

    gl = 10.  # Leak Conductance   [nS]
    El = -60.  # Resting Potential [mV]
    g_Na = 20. * 1000
    ENa = 50.  # reversal potential (Sodium) [mV]
    g_Kd = 6. * 1000  # K Conductance      [nS]
    EK = -90.  # reversal potential (Potassium) [mV]
    VT = -63.
    Vt = -20.
    # Time constants
    taue = 5.  # Excitatory synaptic time constant [ms]
    taui = 10.  # Inhibitory synaptic time constant [ms]
    # Reversal potentials
    Ee = 0.  # Excitatory reversal potential (mV)
    Ei = -80.  # Inhibitory reversal potential (Potassium) [mV]
    # excitatory synaptic weight
    we = 6.0 * np.sqrt(3200) / np.sqrt(num_exc)  # excitatory synaptic conductance [nS]
    # inhibitory synaptic weight
    wi = 67.0 * np.sqrt(800) / np.sqrt(num_inh) # inhibitory synaptic conductance [nS]
    
    inf = 0.05

    neu_ST = bp.types.NeuState('V', 'm', 'n', 'h', 'sp', 'ge', 'gi')

    @bp.integrate
    def int_ge(ge, t):
        return - ge / taue


    @bp.integrate
    def int_gi(gi, t):
        return - gi / taui


    @bp.integrate
    def int_m(m, t, V):
        a = 13.0 - V + VT
        b = V - VT - 40.0
        m_alpha = 0.32 * a / (exp(a / 4.) - 1.)
        m_beta = 0.28 * b / (exp(b / 5.) - 1.)
        dmdt = (m_alpha * (1. - m) - m_beta * m)
        return dmdt
    
    @bp.integrate
    def int_m_zeroa(m, t, V):
        b = V - VT - 40.0
        m_alpha = 0.32
        m_beta = 0.28 * b / (exp(b / 5.) - 1.)
        dmdt = (m_alpha * (1. - m) - m_beta * m)
        return dmdt
    
    @bp.integrate
    def int_m_zerob(m, t, V):
        a = 13.0 - V + VT
        m_alpha = 0.32 * a / (exp(a / 4.) - 1.)
        m_beta = 0.28
        dmdt = (m_alpha * (1. - m) - m_beta * m)
        return dmdt

    @bp.integrate
    def int_h(h, t, V):
        h_alpha = 0.128 * exp((17. - V + VT) / 18.)
        h_beta = 4. / (1. + exp(-(V - VT - 40.) / 5.))
        dhdt = (h_alpha * (1. - h) - h_beta * h)
        return dhdt


    @bp.integrate
    def int_n(n, t, V):
        c = 15. - V + VT
        n_alpha = 0.032 * c / (exp(c / 5.) - 1.)
        n_beta = .5 * exp((10. - V + VT) / 40.)
        dndt = (n_alpha * (1. - n) - n_beta * n)
        return dndt

    @bp.integrate
    def int_n_zero(n, t, V):
        n_alpha = 0.032
        n_beta = .5 * exp((10. - V + VT) / 40.)
        dndt = (n_alpha * (1. - n) - n_beta * n)
        return dndt


    @bp.integrate
    def int_V(V, t, m, h, n, ge, gi):
        g_na_ = g_Na * (m * m * m) * h
        g_kd_ = g_Kd * (n * n * n * n)
        dvdt = (gl * (El - V) + ge * (Ee - V) + gi * (Ei - V) -
                g_na_ * (V - ENa) - g_kd_ * (V - EK)) / Cm
        return dvdt

    def neu_update(ST, _t):
        ST['ge'] = int_ge(ST['ge'], _t)
        ST['gi'] = int_gi(ST['gi'], _t)
        if abs(ST['V'] - (40.0 + VT)) < inf:
            ST['m'] = int_m_zerob(ST['m'], _t, ST['V'])
        elif abs(ST['V'] - (13.0 + VT)) < inf:
            ST['m'] = int_m_zeroa(ST['m'], _t, ST['V'])
        else:
            ST['m'] = int_m(ST['m'], _t, ST['V'])
        ST['h'] = int_h(ST['h'], _t, ST['V'])
        if abs(ST['V'] - (15.0 + VT)) > inf:
            ST['n'] = int_n(ST['n'], _t, ST['V'])
        else:
            ST['n'] = int_n_zero(ST['n'], _t, ST['V'])
        V = int_V(ST['V'], _t, ST['m'], ST['h'], ST['n'], ST['ge'], ST['gi'])
        sp = ST['V'] < Vt and V >= Vt
        ST['sp'] = sp
        ST['V'] = V

    neuron = bp.NeuType(name='CUBA-HH', ST=neu_ST, steps=neu_update, mode='scalar')

    requires_exc = {
        'pre': bp.types.NeuState(['sp'], help='Pre-synaptic neuron state must have "spike" item.'),
        'post': bp.types.NeuState(['ge'], help='Post-synaptic neuron state must have "V" and "input" item.')
    }

    def update_syn_exc(ST, pre, post):
        if pre['sp']:
            post['ge'] += we
    
    exc_syn = bp.SynType(name = 'exc_syn',
        ST=bp.types.SynState(),
        requires = requires_exc,
        steps = update_syn_exc,
        mode = 'scalar')

    requires_inh = {
        'pre': bp.types.NeuState(['sp'], help='Pre-synaptic neuron state must have "spike" item.'),
        'post': bp.types.NeuState(['gi'], help='Post-synaptic neuron state must have "V" and "input" item.')
    }

    def update_syn_inh(ST, pre, post):
        if pre['sp']:
            post['gi'] -= wi
    
    inh_syn = bp.SynType(name = 'inh_syn',
        ST=bp.types.SynState(),
        requires = requires_inh,
        steps = update_syn_inh,
        mode = 'scalar')

    group = bp.NeuGroup(neuron, geometry = num)
    group.ST['V'] = El + (np.random.randn(num_exc + num_inh) * 5. - 5.)
    group.ST['ge'] = (np.random.randn(num_exc + num_inh) * 1.5 + 4.) * 10.
    group.ST['gi'] = (np.random.randn(num_exc + num_inh) * 12. + 20.) * 10.

    exc_conn = bp.SynConn(exc_syn, pre_group=group[:num_exc], post_group=group,
                        conn=bp.connect.FixedProb(prob=0.02))

    inh_conn = bp.SynConn(inh_syn, pre_group=group[num_exc:], post_group=group,
                        conn=bp.connect.FixedProb(prob=0.02))

    net = bp.Network(group, exc_conn, inh_conn)
    ed_build = time.time()

    st_run = time.time()
    net.run(duration=1000.0)
    ed_run = time.time()

    build_time = float(ed_build - st_build)
    run_time = float(ed_run - st_run)

    print('BuildT:{:.2f}s, RunT:{:.2f}s'.format(build_time, run_time))
    return run_time, build_time

if __name__ == '__main__':
    device_list = ['cpu', 'multi-cpu', 'gpu']
    max_num = 18
    min_num = 5
    repeat = 3
    num_list = np.zeros(max_num - min_num + 1)
    for i in range(max_num - min_num + 1):
        num_list[i] = int(2 ** (i + min_num))

    run_t_lif = np.zeros((num_list.size, len(device_list)))
    build_t_lif = np.zeros((num_list.size, len(device_list)))

    run_t_hh = np.zeros((num_list.size, len(device_list)))
    build_t_hh = np.zeros((num_list.size, len(device_list)))

    with open('brainpy_runtime_lif.csv', 'wt', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(device_list)
        for i in range(num_list.size):
            for j in range(len(device_list)):
                for k in range(repeat):
                    rt_lif, bt_lif = test_lif(num = num_list[i], device = device_list[j])
                    run_t_lif[i][j] += rt_lif
                    build_t_lif[i][j] += bt_lif
                run_t_lif[i][j] /= repeat
                build_t_lif[i][j] /= repeat
            writer.writerow(run_t_lif[i])
    
    with open('brainpy_runtime_hh.csv', 'wt', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(device_list)
        for i in range(num_list.size):
            for j in range(len(device_list)):
                for k in range(repeat):
                    rt_hh , bt_hh = test_hh(num = num_list[i], device = device_list[j])
                    run_t_hh[i][j] += rt_hh
                    build_t_hh[i][j] += bt_hh
                run_t_hh[i][j] /= repeat
                build_t_hh[i][j] /= repeat
            writer.writerow(run_t_hh[i])
