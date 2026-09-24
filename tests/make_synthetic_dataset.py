"""Builds a small ToN-IoT-shaped raw directory (PCAPs + Network_dataset_1.csv) for
PIPELINE TESTS ONLY. It is never a data source for any reported result.

    python tests/make_synthetic_dataset.py /tmp/synth_toniot
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic import http_session, udp_packet, write_pcap  # noqa: E402

REQUESTS = {
    'normal': [b"GET /index.html HTTP/1.1\r\nHost: shop\r\n\r\n", b"GET /img/logo.png HTTP/1.1\r\n\r\n"],
    'xss': [b"GET /search?q=<ScRiPt>alert(document.cookie)</sCrIpT> HTTP/1.1\r\n\r\n",
            b"GET /c?x=<img src=x onerror=alert(1)> HTTP/1.1\r\n\r\n"],
    'injection': [b"GET /item.php?id=1' OR '1'='1'-- HTTP/1.1\r\n\r\n",
                  b"GET /u?id=1 UNION SELECT user,pass FROM users HTTP/1.1\r\n\r\n"],
    'password': [b"POST /login HTTP/1.1\r\n\r\nuser=admin&pass=123456", b"POST /login HTTP/1.1\r\n\r\nuser=root&pass=toor"],
    'scanning': [b"HEAD / HTTP/1.0\r\n\r\n"],
}


def build(root: str, n_blocks: int = 48, sessions_per_block: int = 14, seed: int = 0):
    rng = np.random.default_rng(seed)
    classes = list(REQUESTS)
    rows, packets = [], []
    t0 = 1554200000.0
    for b in range(n_blocks):
        cls_block = classes[b % len(classes)]
        for s in range(sessions_per_block):
            cls = cls_block if rng.random() < 0.7 else 'normal'
            client = f"192.168.1.{10 + (b * 7 + s) % 40}" if cls == 'normal' else f"10.9.0.{b % 4 + 1}"
            cport = 20000 + b * 100 + s
            ts = t0 + b * 300 + s * 5 + rng.random()
            req = REQUESTS[cls][int(rng.integers(len(REQUESTS[cls])))]
            packets += http_session(client, cport, '192.168.1.200', ts, req)
            rows.append({'ts': int(ts) if rng.random() < 0 else ts, 'src_ip': client, 'src_port': cport,
                         'dst_ip': '192.168.1.200', 'dst_port': 80, 'proto': 'tcp', 'duration': 0.021,
                         'label': int(cls != 'normal'), 'type': cls})
        # one encrypted-looking UDP flow per block (no readable payload)
        ts = t0 + b * 300 + 200
        packets.append((ts, udp_packet('192.168.1.50', 4433, '192.168.1.200', 443,
                                       b'\x17\x03\x03\x00\x40' + rng.integers(0, 256, 64, dtype=np.uint8).tobytes())))
        rows.append({'ts': ts, 'src_ip': '192.168.1.50', 'src_port': 4433, 'dst_ip': '192.168.1.200', 'dst_port': 443,
                     'proto': 'udp', 'duration': 0.0, 'label': 0, 'type': 'normal'})

    pcap_dir = os.path.join(root, 'Raw_datasets', 'Network_dataset_pcaps', 'capture_a')
    csv_dir = os.path.join(root, 'Processed_datasets', 'Processed_Network_dataset')
    os.makedirs(pcap_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    packets.sort(key=lambda p: p[0])
    half = len(packets) // 2
    write_pcap(os.path.join(pcap_dir, 'normal_1.pcap'), packets[:half])
    write_pcap(os.path.join(pcap_dir, 'normal_2.pcap'), packets[half:])
    pd.DataFrame(rows).to_csv(os.path.join(csv_dir, 'Network_dataset_1.csv'), index=False)
    return root


if __name__ == '__main__':
    build(sys.argv[1])
    print(f"synthetic dataset written to {sys.argv[1]}")
