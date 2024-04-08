

import pickle
import dpkt
import numpy as np
from tqdm import trange


max_byte_len = 250
max_flow = 100000

attack_cat = ['FTP-Patator', 'SSH-Patator', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest', 'Heartbleed',
              'Web Attack - Brute Force', 'Web Attack - Sql Injection', 'Web Attack - XSS', 'Infiltration', 'Bot', 'PortScan', 'DDoS', 'benign']

def mask(p):
    p.src = b'\x00\x00\x00\x00'
    p.dst = b'\x00\x00\x00\x00'
    p.sum = 1
    p.id = 0
    p.offset = 0

    if isinstance(p.data, dpkt.tcp.TCP):
        p.data.sport = 0
        p.data.dport = 0
        p.data.seq = 0
        p.data.ack = 0
        p.data.sum = 1

    elif isinstance(p.data, dpkt.udp.UDP):
        p.data.sport = 0
        p.data.dport = 0
        p.data.sum = 1

    return p


def pkt2feature(data,app,k):

    flow_dict = {'train': {}, 'test': {}}
    for p in attack_cat:
        flow_dict['train'][p] = []
        flow_dict['test'][p] = []
        p_keys = list(data[p].keys())
        now_num = -1

        flownum = 0
        if len(data[p]) > max_flow:
            gap = max_flow
        else:
            gap = len(data[p])
        for flow in p_keys:
            if flownum == max_flow:
                break
            now_num = now_num + 1
            all_pkts = []
            pkts = data[p][flow]
            all_pkts.extend(pkts)
            byte = []
            for idx in range(min(len(all_pkts), 3)):
                if len(byte) <= max_byte_len:
                    pkt = mask(all_pkts[idx][0])
                    if idx == 0:
                        raw_byte = pkt.pack()
                    else:
                        raw_byte = pkt.data.pack()
                    for x in range(len(raw_byte)):
                        byte.append(int(raw_byte[x]))
                else:
                    break
            byte.extend([0] * (max_byte_len - len(byte)))
            byte = byte[0:int(max_byte_len)]
            app_now = app[p][flow]
            for x in range(len(app_now)):
                byte.append(int(app_now[x]))
            if now_num in range(k * int(gap * 0.2), (k + 1) * int(gap * 0.2)):
                flow_dict['test'][p].append((byte))
            else:
                flow_dict['train'][p].append((byte))
    return flow_dict


def load_epoch_data(flow_dict, train='train'):
    flow_dict = flow_dict[train]
    x, label = [], []

    for p in attack_cat:
        pkts = flow_dict[p]
        for byte in pkts:
            x.append(byte)
            label.append(attack_cat.index(p))

    return np.array(x), np.array(label)[:, np.newaxis]


if __name__ == '__main__':

    with open('./pro_flows_1_5.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('./pro_flows_1_5_app.pkl', 'rb') as f:
        app = pickle.load(f)

    for i in trange(1, mininterval=5, \
                    desc='  - (Building fold dataset)   ', leave=False):
        flow_dict = pkt2feature(data,app,i)
        with open('./flows_%d_fold_cut.pkl' % i, 'wb') as f:
            pickle.dump(flow_dict, f)
