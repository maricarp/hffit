import sys
import matplotlib.pyplot as plt
import pesummary
from pesummary.io import read
import h5py
import numpy as np
import pickle
from figaro.mixture import DPGMM
from tqdm import tqdm
import os
import re

events_list = ["IGWN-GWTC3p0-v1-GW200225_060421", "IGWN-GWTC3p0-v1-GW200129_065458",
              "IGWN-GWTC3p0-v1-GW200316_215756", "IGWN-GWTC2p1-v2-GW190828_063405",
              "IGWN-GWTC3p0-v1-GW200202_154313", "IGWN-GWTC3p0-v1-GW191204_171526",
              "IGWN-GWTC3p0-v1-GW191216_213338", "IGWN-GWTC3p0-v1-GW200311_115853",
              "IGWN-GWTC3p0-v1-GW191129_134029", "IGWN-GWTC3p0-v1-GW200115_042309",
              "IGWN-GWTC2p1-v2-GW190408_181802", "IGWN-GWTC2p1-v2-GW190728_064510",
              "IGWN-GWTC2p1-v2-GW190814_211039", "IGWN-GWTC2p1-v2-GW190707_093326",
              "IGWN-GWTC2p1-v2-GW190521_074359", "IGWN-GWTC2p1-v2-GW151226_033853",
              "IGWN-GWTC2p1-v2-GW170814_103043", "IGWN-GWTC2p1-v2-GW190512_180714",
              "IGWN-GWTC2p1-v2-GW190828_065509", "IGWN-GWTC2p1-v2-GW190412_053044",
              "IGWN-GWTC2p1-v2-GW170608_020116", "IGWN-GWTC2p1-v2-GW190720_000836",
              "IGWN-GWTC2p1-v2-GW190708_232457", "IGWN-GWTC2p1-v2-GW190630_185205",
              "IGWN-GWTC2p1-v2-GW170104_101158", "IGWN-GWTC2p1-v2-GW190924_021846",
              "IGWN-GWTC2p1-v2-GW150914_095045"]

parameter = "total_mass_source"

check_list = []
event_name_list = []
for n in events_list:
    events_split = re.split("-|_", n)
    event = events_split[3]
    file_name = n + "_PEDataRelease_mixed_nocosmo.h5"

    if event in check_list:
        event_name = event + "B"
    elif event == "GW190521" or event == "GW191204" or event == "GW200311":
        event_name = event + "B"
    elif event == "GW200115":
        event_name = event
    else:
        event_name = event + "A"

    with h5py.File(file_name, "r") as f:
        print("H5 data sets:")
        print(list(f))

    data = read(file_name)
    print("Run labels:")
    print(data.labels)

    samples_dict = data.samples_dict
    posterior_samples = samples_dict["C01:Mixed"]
    # check what parameters are available within posterior_samples
    parameters = posterior_samples.parameters
    # print(parameters[:])

    # see an explanation of every parameter in these samples using the .description method
    param = parameters[29]
    print(f"The definition of {param} is: {param.description}")

    # boundaries of the distribution
    x_min = 0
    x_max = 200
    mix = DPGMM([[x_min, x_max]])
    samples = posterior_samples[parameter]

    for s in tqdm(samples):  # progress bar of passing samples to the mixture
        mix.add_new_point(s)

    rec = mix.build_mixture()
    file_name = os.path.join(event_name,"dpgmm_" + parameter + "_" + event_name + ".p")
    os.makedirs(event_name, exist_ok = True)
    with open(file_name, 'wb') as f:
        pickle.dump(rec, f)

    check_list.append(event)
    event_name_list.append(event_name)

np.savetxt('events_list.txt', event_name_list)
