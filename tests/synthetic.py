"""Tiny synthetic captures for UNIT TESTS ONLY. Never used for any reported result."""

import socket

import dpkt


def _ip(a: str) -> bytes:
    return socket.inet_aton(a)


def tcp_packet(src, sport, dst, dport, flags, seq=0, payload=b'', win=64240):
    tcp = dpkt.tcp.TCP(sport=sport, dport=dport, seq=seq, ack=0, flags=flags, win=win, off=5, data=payload)
    ip = dpkt.ip.IP(src=_ip(src), dst=_ip(dst), p=dpkt.ip.IP_PROTO_TCP, data=tcp, ttl=64)
    ip.len = len(ip)
    eth = dpkt.ethernet.Ethernet(src=b'\x00' * 6, dst=b'\x11' * 6, type=dpkt.ethernet.ETH_TYPE_IP, data=ip)
    return bytes(eth)


def udp_packet(src, sport, dst, dport, payload=b''):
    udp = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
    udp.ulen = len(udp)
    ip = dpkt.ip.IP(src=_ip(src), dst=_ip(dst), p=dpkt.ip.IP_PROTO_UDP, data=udp, ttl=64)
    ip.len = len(ip)
    return bytes(dpkt.ethernet.Ethernet(src=b'\x00' * 6, dst=b'\x11' * 6, type=dpkt.ethernet.ETH_TYPE_IP, data=ip))


def http_session(client, cport, server, t0, request: bytes, response: bytes = b'HTTP/1.1 200 OK\r\n\r\nhello'):
    S, A, P, F = dpkt.tcp.TH_SYN, dpkt.tcp.TH_ACK, dpkt.tcp.TH_PUSH, dpkt.tcp.TH_FIN
    return [
        (t0 + 0.000, tcp_packet(client, cport, server, 80, S, seq=100)),
        (t0 + 0.001, tcp_packet(server, 80, client, cport, S | A, seq=500)),
        (t0 + 0.002, tcp_packet(client, cport, server, 80, A, seq=101)),
        (t0 + 0.003, tcp_packet(client, cport, server, 80, P | A, seq=101, payload=request)),
        (t0 + 0.004, tcp_packet(client, cport, server, 80, P | A, seq=101, payload=request)),  # retransmission
        (t0 + 0.010, tcp_packet(server, 80, client, cport, P | A, seq=501, payload=response)),
        (t0 + 0.020, tcp_packet(client, cport, server, 80, F | A, seq=101 + len(request))),
        (t0 + 0.021, tcp_packet(server, 80, client, cport, F | A, seq=501 + len(response))),
    ]


def write_pcap(path, packets):
    with open(path, 'wb') as f:
        w = dpkt.pcap.Writer(f)
        for ts, buf in sorted(packets, key=lambda p: p[0]):
            w.writepkt(buf, ts=ts)
