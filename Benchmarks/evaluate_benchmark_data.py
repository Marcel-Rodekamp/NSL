import numpy as np
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import re
import matplotlib.pyplot as plt
from collections import OrderedDict

def get_usage_msg():
    msg = "Usage:\n"
    msg += f"python {sys.argv[0]} /path/to/benchmark/data.txt 'benchmark_name' (given by catch2)"

    return msg


def _investigate_xml(fn):
    # I used this function to investigate the data structure of the xml file
    # if you ever need to do this please install xmltodict e.g. with
    # pip install xmltodice --user
    import xmltodict

    # convert xml to orderd dict
    with open(fn,'r') as xml_f:
        data = xmltodict.parse(xml_f.read())

    # open the benchmark results
    for testCase in data['Catch']['Group']['TestCase']['BenchmarkResults']:
        # testCase = testCase['Section'] # if TestCases are are generated with Sections

        #print all possible values
        # print(testCase,'\n')

        print(f"{testCase['@name']}")
        print(f"mean = {testCase['mean']}")
        print(f"var  = {testCase['standardDeviation']}",'\n')

def read_benchmark_from_xml(fn,name_str):

    root_node = ET.parse(fn).getroot()

    out = dict()

    for benchmark in root_node.findall(name_str):
        # print("tag = ", benchmark, benchmark.tag, benchmark.attrib)
        if benchmark.tag == "OverallResult":
            continue

        out[re.findall(r'\d+',benchmark.attrib['name'])[-1]] = {
            'est': float(benchmark.find('mean').get('value')),
            'var': float(benchmark.find('standardDeviation').get('value'))
        }

    return out

if __name__ == "__main__":
    # get the path to the file from argument
    if "-h " in sys.argv:
        print(get_usage_msg())

    try:
        data_filename = Path(sys.argv[1]).absolute()
    except(IndexError):
        print(get_usage_msg())
        sys.exit(1)

    print(f"Processing Benchmark data from file: {data_filename}\n")

    data_dict = read_benchmark_from_xml(data_filename,"Group/TestCase/")

    Ns        = np.array([int(x) for x in data_dict.keys()])
    time_ests = np.array([x['est']/1000 for x in data_dict.values()])
    time_errs = np.sqrt([x['var'] for x in data_dict.values()])/1000

    fig,ax = plt.subplots(2)
    ax[0].errorbar(Ns,time_ests,yerr=time_errs,label="1D Constructor")
    ax[0].set_xscale('log')
    ax[0].set_xlabel("Number of Elements")
    ax[0].set_ylabel("Time [\u00b5s]")
    ax[0].legend()
    ax[1].errorbar(Ns,Ns*(8*1.25e-7)/(time_ests*1e-6),yerr= (Ns*(8*1.25e-7)/(time_ests**2*1e-6))*time_errs*1e-6,label="1D Constructor")
    ax[1].set_xscale('log')
    ax[1].set_xlabel("Number of Elements")
    ax[1].set_ylabel("Bandwidth [Mb/s]")
    ax[1].legend()

    fig.suptitle("Kokkos Tensor 1D Constructor")
    fig.tight_layout()
    fig.savefig("Benchmark.pdf")