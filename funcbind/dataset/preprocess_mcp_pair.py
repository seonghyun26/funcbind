import argparse
import gc
import os
import torch

from openbabel import pybel
import numpy as np
from tqdm import tqdm

from funcbind.utils.constants import ELEMENT_2_HASH_FUNCBIND


# from funcbind.utils.constants import ATOMIC_NUM_TO_ELEMENT
ATOMIC_NUM_TO_ELEMENT = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    12: "Mg",
    15: "P",
    16: "S",
    17: "Cl",
    20: "Ca",
    25: "Mn",
    35: "Br"
}

DATA_DIR = "data/mcpp_dataset/"

CLUSTERS = [['1sle', '1vwi', '1vwk', '1vwj', '1vwl', '1hxl', '1hxz', '1vwh'],
            ['1sld', '1vwn', '1vwo', '1vwm', '1vwb', '1vwc', '1vwp', '1vwg', '1vwq', '1vwd', '1vwe', '1vwr'], ['1tk2'],
            ['1mik', '1cwf', '1cyn', '1cwa', '1cwb', '1cwc', '1cwo', '1cwm', '1cwl', '1cwi', '1cwk', '1cwj', '1qng', '1bck',
             '2wfj', '2x7k', '3eov', '4dgc', '4hy7', '6hmz', '7pcj', '4ipz', '6x4o', '6x4p', '6x4q', '6x4n', '6x4m', '1cwh'],
            ['1waw', '1wb0'], ['1ikf', '2ck0', '3g5v', '3g5y', '4ioi', '5u6a', '5u5m', '5t1m', '5u5f', '6bae'],
            ['1smf', '1g9i', '1sfi', '1yf4', '4xoj', '4k8y', '4hgc', '4ktu', '4kel', '4k1e', '6bvh', '6vxy', '6u22',
            '4abi', '4kts', '4abj', '1tyn'], ['1qnh', '2z6w', '3pmp', '4jjm'], ['1xq7', '2poy', '4tot', '5hsv'], ['1npo'],
            ['1m63'], ['1c5f'], ['1jk4'], ['1e4w', '3f58', '3pp4', '2f58'], ['1ebp'], ['1h0g', '1h0i'], ['1urc'], ['1hqq'],
            ['1pcg', '2yja', '5wgq', '5dx3', '5dxe', '5wgd', '2yjd', '5hyr', '5dxb'], ['1w9u', '1w9v'], ['1vpp', '6z13', '6zbr',
            '6zcd', '6z3f'], ['1bzh'], ['1l5g'], ['1jbu'], ['1mf8', '6tz7'], ['2uz6', '2byp', '2br8', '5co5', '5jme', '4ez1'],
            ['2nwn', '3ox7', '3oy5', '3oy6', '3m61', '4zks', '4x0w', '4x1s', '4zhm', '4x1p', '4x1q', '4zhl', '4gly', '4os2',
             '4os1', '4zkn', '4x1n', '5wxr', '5wxq', '5wxf', '5wxo', '6xvd', '3qn7', '4x1r', '4jk5', '4jk6', '4os4', '4os5',
             '4os7', '4mnw', '4mnx', '4os6'], ['2esl', '2x2c', '2rmc'], ['2oju'], ['2cv3', '1qr3', '4gvu', '1okx'],
             ['2c9t', '7egr'], ['2rmb', '2rma', '3odi', '3odl'], ['2axi', '6t2f', '6t2d', '4n5t', '5afg'], ['3bo7'], ['3wmg'],
             ['3pqz', '4x6s', '5u1q', '5u06', '5eeq', '5eel', '5tyi'], ['3mt6', '5dkp', '4uog'], ['3avg', '3avf', '3avb', '3avc',
            '3ava', '3avm', '3avl', '3avn', '3av9', '3avk', '3avj', '3avh', '3avi', '3wne', '3wnf', '3wng', '4y1d', '3wnh', '4y1c'],
            ['3p72'], ['3wde', '3wdd', '3wdc', '6cn8', '7aa4', '6pbs'], ['3p8f'], ['3kti', '7fer'], ['3vvr', '3vvs', '3wbn'],
            ['3n7y', '2aob', '2aoa'], ['3n84'], ['4z0d', '4z0f', '4z09', '4z0e'], ['4w50', '4w4z', '5jr2'], ['4yv9'],
            ['4ib5', '5n1v', '6yul', '6q4q'], ['4gux'], ['4gw1', '4gw5', '4m1d', '5ivz', '5euk', '5ir1', '5hyq', '5t1l', '5th2',
            '5t1k', '5id0', '5f88', '5iv2', '5ff6', '6azk', '6ayn', '5id1', '5iop'], ['4ou3'], ['4zqw'], ['4zjx'], ['5vb9', '8cdg'],
            ['5djz', '5dj8', '5djy', '5dvn', '5dvl', '5djd', '5dj0', '5di8', '5dk0', '5dj6', '5u66'], ['5djx', '5djc'],
            ['5i8m', '5nf0', '5ney', '5nes', '6y0v', '6y0u', '5i8x'], ['5eoc'], ['5glh', '3oe0'], ['5xn3', '6dn5', '6dn6'],
            ['5tu6'], ['5o4y', '6nnv'], ['5ly1', '5ly2'], ['5n99'], ['5kez'], ['5c3g', '2yq6'], ['5b4w'],
            ['5kgn', '7kng', '7knf', '7tl7'], ['5h5r', '5h5s', '5h5q'], ['5jzu', '6rk9'], ['5grd', '5grg'], ['5lso'],
            ['5o45', '6pv9', '7oun'], ['5xco'], ['6hza', '6hzb', '6hzd', '6hle', '6hzc'], ['6c1q', '6c1r'],
            ['6c1d', '7pmb', '7pmc', '7pma', '7pm7', '7pm6', '7plw', '7pmd', '7plv', '7plt', '7pmg', '7pmf', '7bti', '7p1g',
            '7ad9', '7tpt', '7plx', '7pmj', '7pmh', '7pmi', '7pml', '7pm8', '7pm9', '7pm5', '7ad8', '7p2g'], ['6b67'],
            ['6xs5', '6xsa', '6xs7', '6xs8'], ['6u4a', '6u61', '6u8m', '6u72'], ['6dn8', '6dn7'], ['6u6k', '6u8h'], ['6j8e'],
            ['6xci'], ['6d3z', '6d3y', '6d40'], ['6a8n', '4mny'], ['6d3x', '6q1u'], ['6whq', '6who', '6whn', '6wi3'], ['6s35'],
            ['6xbf', '6xbe'], ['6whz'], ['6yw2', '6yw3', '6yw1', '6yw4'], ['6xs9'], ['6ulv'], ['6u8i', '6u71'],
            ['6d8c', '7pme', '7plu', '7pdz'], ['7wj5', '7ryc', '7f55', '7bb7', '7t11', '7bb6', '7piu', '7kh0', '7dw9', '7wic'],
            ['7fep', '5w18', '6pka', '6pmd', '5vz2'], ['7ujd'], ['7n0w', '7n43'], ['7e5e'],
            ['7k2i', '7k2h', '7k2k', '7k2n', '7k2l', '7k2m', '7k2f', '7k2g', '7k2r', '7k2e', '7k2s', '3zgc', '6t7z', '7q5h',
             '6z6a', '7k2o', '7q6q', '7q6s', '7k2q', '7q8r', '7q96', '7k2p'], ['7mix', '7vfu'],
             ['7ei2', '7ehz', '7wmc', '7egu', '7wmt'], ['7bph'], ['7yf6', '1b6j', '1b6l', '1mtr', '1d4l', '1b6p', '3bxs',
            '1b6k', '1b6m', '1z1h', '1cpi', '1z1r', '3bxr', '1d4k'], ['8ryk'], ['8q1n', '4gmb', '5vfc'], ['8ulg'], ['1bxo'],
            ['1w0e'], ['1jd2', '2gpl', '8rhl'], ['1bm2'], ['2bcd', '2bdx'], ['2a68', '6fbv', '4oir', '4mq9', '2a69'], ['3tdz'],
            ['3e7a', '6obu', '3egh', '1fjm'], ['3rif'], ['3v3b', '6y4q', '7nus', '6h22'], ['3rqd'], ['3mgn'], ['4wvt'],
            ['4n7y', '4n84', '5j31', '6rlz', '5jm4'], ['4nec'], ['4i80'], ['4l3o'], ['4dri', '7aou', '7awf'], ['4cli'],
            ['4wh8', '4ktc', '6p6l', '3sud', '5epy'], ['4dpf', '4gmi'], ['5vbl'], ['5mwj'], ['5whh'],
            ['5w6u', '5w5s', '5w6i', '5w5u', '5w6t'], ['5bmm'], ['5bxo'], ['5zjz', '5zml', '5zk9', '5zk5'], ['5nxq'],
            ['5tg2', '5tg1', '6bid'], ['5vk0'], ['5mev', '8av9', '5mes'], ['5vi6'], ['5w8f', '5w89'], ['5zk7'], ['5v2p', '5v2q'],
            ['5agv', '5ah4', '5agu', '6dly'], ['5lrj', '5lrg', '5lrk'], ['5ooc'], ['5jh7', '7daf'], ['5dg6', '6bib', '6bic'],
            ['6x3j'], ['6dm6'], ['6z59', '6z5d', '6z5b', '6z58', '6z5e', '6z5a', '6z5c'], ['6wsj'], ['6exv'],
            ['6z4z', '6z50', '6z51', '6z53', '6z54'], ['6p81'], ['6z48', '8asf', '6t7h', '1tmb', '6gwe'], ['6ptv'], ['6y0j'],
            ['6q38'], ['6xib', '6xid', '6xif', '7kfa', '7s5g', '7s5h', '6xic', '6xie'], ['6hy7'], ['6c0y'],
            ['7o2m', '7oc2', '7vli'], ['7p0i', '8ax6', '8ax7', '7p0f', '8ax5'], ['7ar4'], ['7d5i'], ['7mso', '6ax4', '4x9w', '4x9r'],
            ['7rov'], ['7f0f', '7f0c'], ['7rnw'], ['8eqi'], ['8ase'], ['8j5x', '8j5w', '8j61', '8j63', '7xaf'],
            ['8rq8'], ['8ytd'], ['3dv5', '3dv1'], ['4wvu'], ['4jna'], ['7mx1', '4x9v'], ['6qbg'], ['7qgp'], ['6zrd'],
            ['7z4s'], ['6nzt', '3kee'], ['7quf', '7que'], ['4dru'], ['5w6r'], ['6sis'], ['5ah2'], ['3s04', '1t7d', '3iiq'],
            ['4twt'], ['5y7w'], ['1kmh'], ['6z52', '6z55'], ['2vdn'], ['5cs2'], ['4q2k'], ['6ptr'], ['4i5l'], ['3rul'], ['5n1z'],
            ['7qpb'], ['6wnk'], ['6j67'], ['7tb1']]

CONF_DICT = conf_dict = {'5zml': 1, '1sle': 502, '7pmb': 869, '6z13': 891, '4wvt': 1, '6hza': 503, '6t2f': 1, '4z0d': 501, '3tdz': 1, '1b6j': 1, '4zks': 503, '8ytd': 1,
            '3dv5': 1, '5u1q': 933, '7awf': 1, '6bvh': 501, '3odl': 1, '7fer': 502, '5i8x': 1, '1b6k': 1, '4z0e': 1, '6c1q': 502, '5vb9': 501, '5jr2': 181,
            '7pm5': 1, '1sld': 502, '4wvu': 1, '7pmc': 597, '3wbn': 1, '4n7y': 1, '5djz': 426, '5djx': 501, '4w50': 605, '7pma': 501, '6hzb': 506, '7wj5': 169,
            '1tk2': 501, '7pm7': 501, '8j5w': 1, '4yv9': 812, '6c1d': 502, '7pcj': 502, '7aou': 1, '5i8m': 501, '7fep': 521, '5eoc': 501, '7tl7': 1, '5u06': 503,
            '6nnv': 763, '5ivz': 501, '6b67': 502, '2yja': 1, '4jna': 1, '4z0f': 501, '5nf0': 1, '7mx1': 1, '3egh': 1, '6c1r': 501, '5vbl': 1, '6t2d': 1, '7pm6': 501,
            '6hzc': 1, '6qbg': 1, '5hsv': 1, '4n84': 1, '5dj8': 501, '5djy': 501, '7ujd': 255, '7plw': 518, '3bo7': 502, '7n0w': 501, '7pmd': 939, '6xs5': 501,
            '1mik': 807, '7ryc': 568, '5glh': 502, '5n1v': 1, '5ney': 1, '7qgp': 1, '1b6l': 1, '5wgd': 1, '4x0w': 502, '3sud': 1, '4x1s': 501, '4oir': 1,
            '1waw': 501, '2bdx': 1, '1ikf': 503, '4x1r': 1, '8eqi': 1, '2yjd': 1, '6u4a': 505, '1b6m': 1, '6rk9': 197, '5xn3': 292, '8ase': 1, '7pme': 301,
            '1smf': 501, '6dn8': 542, '7plv': 886, '6zrd': 1, '3wmg': 502, '1qnh': 501, '5co5': 668, '6x3j': 1, '2z6w': 502, '6hmz': 310, '7plt': 542, '5tu6': 501,
              '7pmg': 501, '6hzd': 501, '5o4y': 502, '3pqz': 502, '6dm6': 1, '5mwj': 1, '1xq7': 892, '5whh': 1, '5euk': 516, '4zhm': 501, '6bae': 501, '6u6k': 501,
              '2wfj': 505, '7z4s': 1, '4x1p': 795, '1bxo': 1, '4x1q': 296, '3odi': 1, '3dv1': 1, '4zhl': 501, '5wgq': 1, '5lrg': 1, '6nzt': 1, '6xsa': 755,
              '6xs7': 501, '6j8e': 501, '7pmf': 501, '8asf': 1, '4uog': 1, '7plu': 501, '1z1h': 1, '6q4q': 1, '7e5e': 501, '4n5t': 1, '3wnh': 1, '4nec': 1,
              '6obu': 1, '7k2i': 501, '5zk9': 1, '1vwh': 1, '6z58': 1, '6azk': 501, '6vxy': 501, '5ly1': 502, '7o2m': 1, '2uz6': 502, '5n99': 501, '5w6u': 1,
              '5bmm': 1, '5u66': 1, '8ryk': 502, '7mix': 155, '5u6a': 502, '4abi': 1, '3mt6': 502, '5u5m': 502, '7quf': 1, '4x6s': 502, '6y0v': 1, '6u22': 501,
              '5w6t': 1, '1npo': 501, '3ox7': 514, '6t7h': 1, '6ax4': 1, '5ah4': 1, '6z59': 1, '1vwi': 502, '5jme': 504, '1mtr': 1, '7k2h': 501, '1cwf': 501,
              '4gmb': 1, '4i80': 1, '6z6a': 1, '3eov': 502, '1m63': 502, '7bti': 502, '1vwk': 501, '7pdz': 207, '1c5f': 503, '5ly2': 456, '3avg': 501, '6wsj': 1,
              '4tot': 1, '7f55': 1001, '5bxo': 1, '4dgc': 501, '4dru': 1, '5ir1': 502, '1w0e': 1, '6pbs': 1, '4abj': 1, '5kez': 881, '6y0u': 1, '6wi3': 1, '7que': 1,
                '7vfu': 382, '6rlz': 1, '3avf': 501, '7p1g': 505, '2f58': 1, '1jk4': 501, '1vwj': 501, '6cn8': 1, '8av9': 1, '1cyn': 501, '3pmp': 502, '6xci': 502,
                '4ib5': 501, '4y1c': 1, '7k2k': 501, '7ei2': 761, '2ck0': 503, '7k2o': 1, '5hyr': 1, '6x4n': 1, '1cwa': 501, '5zjz': 1, '7bb7': 640, '5c3g': 501,
                '8q1n': 501, '5jm4': 1, '1vwn': 501, '1tmb': 1, '8ax6': 1, '5vfc': 1, '5w89': 1, '3avb': 520, '4mnw': 1, '2byp': 502, '2gpl': 1, '5wxr': 501,
                '5eeq': 502, '7t11': 502, '6exv': 1, '7s5h': 1, '4xoj': 502, '5nxq': 1, '1e4w': 500, '7q6q': 1, '5w6r': 1, '1ebp': 502, '3oy5': 501, '3avc': 501,
                '6d3z': 326, '7p0f': 1, '8ax7': 1, '6sis': 1, '1vwo': 548, '6z4z': 1, '1qr3': 1, '5ah2': 1, '7nus': 1, '7bb6': 716, '5tg2': 1, '6h22': 1, '6x4o': 1,
                '7kfa': 1, '6a8n': 502, '7k2n': 501, '7k2l': 501, '4y1d': 505, '6x4m': 1, '1cwb': 501, '5hyq': 501, '1h0g': 498, '3p72': 501, '6tz7': 1, '3g5v': 501,
                '1vwm': 501, '8ax5': 1, '6d3x': 501, '3ava': 878, '4jjm': 1205, '3s04': 1, '7q5h': 1, '7q6s': 1, '6p81': 1, '4twt': 1, '6whq': 501, '1urc': 501,
                '6p6l': 1, '5wxq': 501, '5b4w': 502, '5wxf': 501, '3oy6': 527, '3m61': 501, '6t7z': 1, '6d3y': 501, '1t7d': 1, '6s35': 406, '7wmc': 1, '5vk0': 1,
                '7wmt': 1, '6z48': 1, '1vwl': 501, '5tg1': 1, '1cwc': 606, '7k2m': 502, '1fjm': 1, '6xbf': 501, '5t1m': 501, '3wde': 501, '6z51': 1, '4l3o': 1,
                '4gvu': 1, '6ayn': 502, '4k8y': 502, '3avm': 502, '5y7w': 1, '4mnx': 1, '5kgn': 1589, '2nwn': 501, '7s5g': 1, '4dri': 1, '4mny': 1, '1kmh': 1,
                '3avl': 294, '4cli': 1, '2esl': 241, '5dxb': 1, '7p0i': 1, '4gux': 506, '6zbr': 1143, '7n43': 838, '6z50': 1, '4gw1': 501, '7egr': 502, '3wdd': 501,
                '5t1l': 501, '4wh8': 1, '1cwo': 501, '3p8f': 895, '5th2': 722, '4gmi': 1, '5mev': 1, '6xbe': 503, '1cwm': 501, '4kts': 1, '3g5y': 501, '1vwb': 501,
                '6z52': 1, '5vi6': 1, '6z5d': 1, '2vdn': 1, '1hqq': 958, '6ptv': 1, '3avn': 501, '6bib': 1, '7ad9': 511, '5w5s': 1, '6y0j': 1, '7ar4': 1, '4x9w': 1,
                '5u5f': 501, '2bcd': 1, '7daf': 1, '4x9v': 1, '7d5i': 1, '1g9i': 501, '4ou3': 501, '7ad8': 502, '5w6i': 1, '6bic': 1, '3av9': 883, '6z5e': 1,
                '7piu': 502, '6z53': 1, '1vwc': 501, '1jd2': 1, '7kh0': 501, '6q38': 1, '1h0i': 501, '4gly': 486, '1cwl': 502, '1cwh': 1, '7k2q': 1, '6x4p': 1,
                '7ehz': 502, '7k2f': 501, '7egu': 1, '4ioi': 501, '6q1u': 501, '1tyn': 1, '3wdc': 503, '5t1k': 501, '1vwp': 502, '2x2c': 501, '5cs2': 1, '6z5a': 1,
                '5h5r': 501, '5dvn': 501, '5jzu': 501, '1vwg': 502, '5dx3': 1, '3e7a': 1, '2oju': 502, '5dxe': 1, '5afg': 1, '5w8f': 1, '6gwe': 1, '3avk': 501,
                '5epy': 1, '7dw9': 591, '5id0': 501, '7q8r': 1, '4jk5': 1, '4dpf': 1, '5grd': 501, '4q2k': 1, '4x9r': 1, '4hgc': 502, '6whz': 501, '7mso': 1,
                '6ptr': 1, '5id1': 1, '7q96': 1, '3avj': 501, '6xvd': 811, '5agu': 1, '7p2g': 501, '5h5s': 382, '1vwq': 502, '3kti': 502, '5zk7': 1, '7k2g': 501,
                '6x4q': 1, '4i5l': 1, '7k2p': 1, '1cwi': 502, '7k2r': 501, '1cwk': 501, '7k2e': 501, '4ktu': 900, '5zk5': 1, '5h5q': 502, '5f88': 390, '6z5b': 1,
                '4gw5': 733, '6z54': 1, '1vwd': 501, '1sfi': 501, '5v2q': 1, '7tpt': 502, '3avh': 501, '3vvr': 502, '6bid': 1, '6pv9': 501, '4jk6': 1, '5w5u': 1,
                '5grg': 501, '6who': 501, '1yf4': 1013, '5wxo': 306, '5eel': 523, '1pcg': 501, '6whn': 502, '6pmd': 1, '3vvs': 353, '3avi': 501, '5v2p': 1, '5agv': 1,
                '1vwe': 501, '6zcd': 502, '6z55': 1, '1vwr': 501, '5dvl': 501, '6z5c': 1, '4ktc': 1, '4ez1': 1, '5mes': 1, '3rul': 1, '1cwj': 501, '7k2s': 501,
                '3wne': 501, '1cpi': 1, '7rov': 1, '6fbv': 1, '6hle': 1, '6xib': 1, '7kng': 602, '6dn6': 1, '7plx': 501, '1z1r': 1, '7bph': 500, '7oc2': 1,
                '4hy7': 501, '7vli': 1, '4m1d': 997, '5lrj': 1, '6yul': 1, '6u61': 503, '4os2': 501, '7f0f': 1, '6yw2': 501, '3f58': 502, '2yq6': 1, '4mq9': 1,
                '1w9u': 501, '6yw3': 642, '7aa4': 1, '5lrk': 1, '5lso': 502, '6u8m': 501, '2poy': 501, '8ulg': 828, '5ooc': 1, '1d4l': 1, '3kee': 1, '7pmj': 501,
                '8j61': 1, '5o45': 501, '4kel': 605, '1vpp': 501, '2x7k': 408, '2cv3': 502, '6dn7': 501, '1okx': 1, '7knf': 671, '6xic': 1, '5djd': 501, '3n7y': 502,
                '1qng': 685, '3wnf': 501, '5dj0': 501, '3iiq': 1, '5jh7': 1, '3bxs': 1, '5xco': 501, '6dn5': 501, '3rif': 1, '8j63': 1, '7pmh': 760, '6xs9': 501,
                '2a69': 1, '4ipz': 1, '5n1z': 1, '5iop': 1, '6yw1': 501, '4os1': 502, '4zkn': 585, '1bck': 502, '7qpb': 1, '4zqw': 643, '1bzh': 502, '5iv2': 501,
                '3qn7': 1, '6pka': 1, '1w9v': 501, '6wnk': 1, '5w18': 1, '2br8': 258, '4z09': 501, '6ulv': 501, '6d40': 501, '6xs8': 501, '2a68': 1, '6z3f': 980,
                '7pmi': 501, '6j67': 1, '3bxr': 1, '5dg6': 1, '3n84': 501, '2c9t': 625, '3wng': 501, '5dkp': 501, '5djc': 415, '6xid': 1, '6dly': 1, '2aob': 1,
                '7wic': 334, '5tyi': 1, '1d4k': 1, '4os4': 1, '6yw4': 1228, '4zjx': 502, '2rmc': 501, '3oe0': 1, '7oun': 494, '7tb1': 1, '3v3b': 1, '2rmb': 600,
                '4os5': 1, '6u72': 502, '1l5g': 501, '1bm2': 1, '7xaf': 1, '8cdg': 1, '1jbu': 518, '4k1e': 828, '7pml': 501, '3pp4': 501, '6xie': 1, '3rqd': 1,
                '5di8': 330, '5dk0': 502, '5j31': 1, '7rnw': 1, '5dj6': 501, '5vz2': 1, '4w4z': 504, '1mf8': 1002, '2aoa': 1, '6hy7': 1, '1hxl': 963, '8j5x': 1,
                '7yf6': 502, '7pm8': 816, '6u8i': 501, '5nes': 1, '7f0c': 1, '4os7': 1, '4x1n': 501, '1wb0': 501, '8rq8': 1, '6y4q': 1, '2rma': 502, '1b6p': 1,
                '4os6': 1, '6u71': 501, '3mgn': 1, '3zgc': 1, '6u8h': 501, '8rhl': 1, '6c0y': 1, '5ff6': 501, '7pm9': 622, '6d8c': 502, '1hxz': 951, '2axi': 502,
                '6xif': 1}


def split_data(data_dir):
    count_per_cluster = []
    for cluster in CLUSTERS:
        cluster_count = 0
        for pdb in cluster:
            for name,count in CONF_DICT.items():
                if pdb == name:
                    cluster_count += count
                    break
        count_per_cluster.append(cluster_count)

    print('count per cluster: ',count_per_cluster)
    print("all counts: ",sum(count_per_cluster))

    train_counts, train_clusters = [], []
    testval_counts, testval_clusters = [], []

    for index, count in enumerate(count_per_cluster):
        if count > 100:
            train_counts.append(count)
            train_clusters.append(CLUSTERS[index])
        else:
            testval_counts.append(count)
            testval_clusters.append(CLUSTERS[index])

    train_clusters = [item for sublist in train_clusters for item in sublist]
    testval_clusters = [item for sublist in testval_clusters for item in sublist]

    # split train clusters into train and val
    testval_clusters = np.array(testval_clusters)
    np.random.seed(1234)
    np.random.shuffle(testval_clusters)
    test_clusters = testval_clusters[-100:]
    val_clusters = testval_clusters[:-100]

    print("train counts: ",len(train_clusters))
    print("val counts: ",len(val_clusters))
    print("test counts: ",len(test_clusters))

    # add data_dir to the clusters
    train_clusters = [os.path.join(data_dir, cluster) for cluster in train_clusters]
    val_clusters = [os.path.join(data_dir, cluster) for cluster in val_clusters]
    test_clusters = [os.path.join(data_dir, cluster) for cluster in test_clusters]
    return {"train": train_clusters, "val": val_clusters, "test": test_clusters}


def get_conformation(mol, mode="numpy"):
    atom_types = []
    coords = []
    for atom in mol.atoms:
        if atom.atomicnum == 1:  # ignore hydrogen atoms
            continue
        coords.append(atom.coords)
        atom_types.append(ATOMIC_NUM_TO_ELEMENT[atom.atomicnum])
    if mode == "numpy":
        coords = np.array(coords, dtype=np.float32)
        atoms_channel = np.array([ELEMENT_2_HASH_FUNCBIND[el] for el in atom_types]).astype(np.byte)
    else:
        coords = torch.tensor(coords, dtype=torch.float32)
        atoms_channel = torch.tensor([ELEMENT_2_HASH_FUNCBIND[el] for el in atom_types])

    conf = {
        "coords": coords,
        "atoms_channel": atoms_channel,
    }
    return conf


def filter_atoms_by_distance_mask(mcp: dict, protein:dict, max_dim: float = 20,):
    # assuming we are modeling a box of siez 128 at resolution .25A
    # going from -16A to 16A. We chose max_dim=4 for an extra buffer
    # get center mcp
    center_mcp = mcp["coords"].mean(axis=0)

    # center mcp and protein
    mcp["coords"] = mcp["coords"] - center_mcp
    protein["coords"] = protein["coords"] - center_mcp

    # mask mcp and protein and remove atoms outside of max_dim
    mask = np.logical_or(mcp["coords"] < -max_dim, mcp["coords"] > max_dim)
    mask = mask.any(axis=1)
    mcp['coords'] = mcp['coords'][~mask]
    mcp['atoms_channel'] = np.array(mcp['atoms_channel'])[~mask]#.tolist()

    mask = np.logical_or(protein["coords"] < -max_dim, protein["coords"] > max_dim)
    mask = mask.any(axis=1)
    protein['coords'] = protein['coords'][~mask]
    protein['atoms_channel'] = np.array(protein['atoms_channel'])[~mask]#.tolist()

    # return to original coordinates
    mcp["coords"] = mcp["coords"] + center_mcp
    protein["coords"] = protein["coords"] + center_mcp

    return mcp, protein


def preprocess_mcp_dataset(data_split, save_dir, split="train"):
    data = {}
    data_split = sorted(data_split)
    for idx, target_path in tqdm(enumerate(data_split)):

        mcp_confs = []
        mcp_paths = []
        prot_confs = []
        prot_paths = []
        has_error= False
        for i, fname in enumerate(os.listdir(target_path)):
            # print(">>", i, fname)
            if has_error:
                has_error = False
                continue
            try:
                if "_mut" in fname and "MCP" in fname:
                    # read the mutants MCPs/Prtoteins
                    mcp_mut_path = os.path.join(target_path, fname)
                    prot_mut_path = mcp_mut_path.replace("MCP.pdb", "protein.pdb")

                    mcp_mut = next(pybel.readfile("pdb", mcp_mut_path))
                    prot_mut = next(pybel.readfile("pdb", prot_mut_path))

                    mcp_mut = get_conformation(mcp_mut)
                    prot_mut = get_conformation(prot_mut)

                    mcp_mut, prot_mut = filter_atoms_by_distance_mask(mcp_mut, prot_mut)
                    mcp_confs.append(mcp_mut)
                    prot_confs.append(prot_mut)
                    mcp_paths.append(mcp_mut_path)
                    prot_paths.append(prot_mut_path)
                elif "CP" in fname:
                    # read crystal MCP/Protein
                    mcp_path = os.path.join(target_path, fname)
                    prot_path = mcp_path.replace("CP.sdf", "protein.pdb")

                    mcp = next(pybel.readfile("sdf", mcp_path))
                    prot = next(pybel.readfile("pdb", prot_path))

                    mcp_conf = get_conformation(mcp)
                    prot_conf = get_conformation(prot)

                    mcp_confs.append(mcp_conf)
                    prot_confs.append(prot_conf)
                    mcp_paths.append(mcp_path)
                    prot_paths.append(prot_path)
            except Exception as e:
                has_error = True
                print(f">> Error reading some files in {target_path} ({i})")
                print(e)
                continue

        target_id = os.path.basename(target_path)
        data[target_id] = {
            "mcp_confs": mcp_confs,
            "mcp_paths": mcp_paths,
            "prot_confs": prot_confs,
            "prot_paths": prot_paths,
        }

        if idx % 100 == 0 and idx > 0:
            torch.save(data, save_dir.replace(".pt", f"_{idx}.pt"))
            del data
            data = {}

    torch.save(data, save_dir)
    del data

    # merge files and/or rename them
    if split == "train":
        # merge all the train splits into one
        os.rename(os.path.join(os.path.dirname(args.data_dir), "train.pt"), os.path.join(os.path.dirname(args.data_dir), "train_500.pt"))
        train_files = [os.path.join(os.path.dirname(args.data_dir), f"train_{i}.pt") for i in range(100, 600, 100)]
        data = {}
        for file in train_files:
            data.update(torch.load(file, weights_only=False))
        torch.save(data, os.path.join(os.path.dirname(args.data_dir), "train_data.pt"))
        for file in train_files:
            os.remove(file)
    else:
        os.rename(os.path.join(os.path.dirname(args.data_dir), f"{split}.pt"), os.path.join(os.path.dirname(args.data_dir), f"{split}_data.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    args = parser.parse_args()

    ob_log_handler = pybel.ob.OBMessageHandler()
    pybel.ob.obErrorLog.StopLogging()

    data_splits = split_data(args.data_dir)
    for split in ["test", "val", "train"]:
        print(f">> preprocessing {split}...")

        save_dir = os.path.join(os.path.dirname(args.data_dir), f"{split}.pt")
        preprocess_mcp_dataset(data_splits[split], save_dir, split=split)
        gc.collect()
