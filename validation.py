import numpy as np
import ROOT
import json

def get_values(h):
    return np.array([h.GetBinContent(i+1) for i in range (h.GetNbinsX())])
def get_variances(h):
    return np.array([h.GetBinError(i+1) for i in range (h.GetNbinsX())])

def match_histos (rdf_hist, coffea_hist, precision):
    rdf_values = get_values(rdf_hist)
    coffea_values = get_values(coffea_hist)
    rdf_values = np.round(rdf_values, precision)
    coffea_values = np.round(coffea_values, precision)
    mask = rdf_values == coffea_values
    return not (False in mask)

def get_deviations(rdf_hist, coffea_hist):
    rdf_values = get_values(rdf_hist)
    coffea_values = get_values(coffea_hist)
    deviations = np.zeros(len(rdf_values))
    for i in range(len(deviations)):
        rdf_value = rdf_values[i]
        coffea_value = coffea_values[i]
        deviations[i] = 0 if round(coffea_value, 5) == round(rdf_value, 5) == 0 else 100*abs(rdf_value-coffea_value)/max(coffea_value, rdf_value)
#         deviations[i] = 100*abs(rdf_value-coffea_value)/coffea_value

    return deviations


def get_mismatched (rdf, coffea):
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
                    deviations = get_deviations(rdf_hist, coffea_hist)
                    i = np.argmax(deviations); dev = deviations[i]
                    variance = coffea_hist.GetBinError(int(i+1))
#                     if deviation > 20:
                    if dev > 0.0001 and "res_up" not in hist_name:
                        print(f"deviation={dev:.2f}%\t-\t{hist_name}")
#                     if "res_up" in hist_name:
#                         print(f"deviation={dev:.2f}% variance/deviation={100*variance/dev:.2f}%\t-\t{hist_name}")
#                         mismatched.append(hist_name)
                else:
                    print(hist_name)
                    raise ValueError('rdf_hist and coffea_hist is Zombie')
    return mismatched

if __name__ == "__main__":
    get_mismatched("rdf.root", "histograms_local.root")