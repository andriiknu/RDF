import numpy as np
import ROOT
import json

def get_values(h):
    return np.array([h.GetBinContent(i+1) for i in range (h.GetNbinsX())])

def match_histos (rdf_hist, coffea_hist, precision):
    rdf_values = get_values(rdf_hist)
    coffea_values = get_values(coffea_hist)
    rdf_values = np.round(rdf_values, precision)
    coffea_values = np.round(coffea_values, precision)
    mask = rdf_values == coffea_values
    return not (False in mask)

def deviations(rdf_hist, coffea_hist):
    rdf_values = get_values(rdf_hist)
    coffea_values = get_values(coffea_hist)
    deviations = np.zeros(len(rdf_values))
    for i in range(len(deviations)):
        rdf_value = rdf_values[i]
        coffea_value = coffea_values[i]
        deviations[i] = 0 if round(coffea_value, 5) == round(rdf_value, 5) == 0 else 100*abs(rdf_value-coffea_value)/max(coffea_value, rdf_value)
#         deviations[i] = 100*abs(rdf_value-coffea_value)/coffea_value

    return deviations


def get_deviations (rdf, coffea):
    mismatched = []
    rdf = ROOT.TFile.Open(rdf)
    coffea = ROOT.TFile.Open(coffea)
    with open('data.json', 'r') as f:
        data = json.load(f)
    for process in data:
        for variation in data[process]:
            for region in data[process][variation]:
                hist_name = f"{region}_{process}_{variation}" if variation != 'nominal' else f"{region}_{process}"
                rdf_hist = rdf.Get(hist_name)
                coffea_hist = coffea.Get(hist_name)
                if (rdf_hist and coffea_hist):
                    deviation = np.max(deviations(rdf_hist, coffea_hist))
#                     if deviation > 20:
                    if deviation > 0.0001 and "res_up" not in hist_name:
                    # if "res_up" in hist_name:
                        print(f"{hist_name} maximum deviation: {deviation} %")
                        mismatched.append(hist_name)
                else:
                    raise ValueError('rdf_hist and coffea_hist is Zombie')
    return mismatched

if __name__ == "__main__":
    get_deviations("rdf.root", "histograms_local.root")