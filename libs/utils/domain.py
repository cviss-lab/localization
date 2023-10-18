import time
import socket
import json
import sys
import logging
from urllib import request 
import yaml
import os

def load_domain(fpath):
    domain = {"domain": None, "token": None}
    if os.path.exists(fpath):        
        with open(fpath, "r") as stream:
            try:
                domain = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(f'Could not read domain paramters in {fpath}')    
    else:
        print(f'No domain paramters were found in {fpath}')        
    return domain

def init_ip_address():
    domain_parameters = load_domain(os.path.join(os.path.abspath('..'),'config/domain.yaml'))
    endpoint = domain_parameters['domain']

    if endpoint is None or endpoint == '' or endpoint == 'localhost':
        ipv4 = get_ipv4_local()
    else:
        ipv4 = '-'

    try:
        ipv6 = request.urlopen('https://v6.ident.me',timeout=5).read().decode('utf8')
    except Exception as e:
        print('\nNo IPv6 address detected! Connect your device to an IPv6 compatible network! (https://test-ipv6.com/)\n')
        
    try:
        parameters = [domain_parameters['domain'],domain_parameters['token'],ipv4,ipv6]
        res=request.urlopen('https://dynv6.com/api/update?zone={}&token={}&ipv4={}&ipv6={}'.format(*parameters)).read()
        if res=='KO':
            print('Error: could not set up DNS server hostname!\n')
    except Exception as e:
        print(e)
        print('Error: could not update DNS server hostname!\n')


    print("\nStarting server on {}".format(endpoint))


def get_ipv4_local():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


