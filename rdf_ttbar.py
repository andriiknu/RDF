#!/usr/bin/env python
# coding: utf-8



import argparse


# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("--ncores", type=int)
parser.add_argument("--nfiles", type=int)
parser.add_argument("--second_disk")
parser.add_argument("--startwith", type=int, help='number of first file')

# Read arguments from command line
args = parser.parse_args()

import json
import ROOT
from ROOT import RDataFrame, TCanvas, THStack
ROOT.EnableImplicitMT(args.ncores or 128)
import numpy as np
# get_ipython().run_line_magic('jsroot', 'on')
print(f'The num of threads = {ROOT.GetThreadPoolSize()}')
verbosity = ROOT.Experimental.RLogScopedVerbosity(ROOT.Detail.RDF.RDFLogChannel(), ROOT.Experimental.ELogLevel.kInfo)
import os
print(os.environ['EXTRA_CLING_ARGS'])

# In[2]:


ROOT.gSystem.CompileMacro("helper.cpp", "kO")


# In[3]:


N_FILES_MAX_PER_SAMPLE = args.nfiles or 1
FILE = f'rdf{N_FILES_MAX_PER_SAMPLE}.root'
START = 0 if not args.startwith else args.startwith

print(f'you are processing {N_FILES_MAX_PER_SAMPLE} files starting from {START}')



# In[16]:


from urllib.request import urlretrieve
import os


class TtbarAnalysis(dict):

    def __init__(self, disk, first=START, n_files_max_per_sample = N_FILES_MAX_PER_SAMPLE, num_bins=25, bin_low = 50, bin_high = 550):
        
        
        self.variations = {} # serves as temporary storage for all histograms produced by VariationsFor
#         self.download_input_data = download_input_data  # set True to download input files
#         self.use_local_data = use_local_data            # set True to use locally placed input files instead of https accessing
        self._nevts_total = {}
        self.disk = disk
        self.first=first
        self.n_files_max_per_sample = n_files_max_per_sample  #the number of files to be processed per sample
        self.total = 0
        self.nfiles = self._construct_fileset() # dictionary assigning file URLs (paths) to each process, variation, and region
        self.num_bins = num_bins
        self.bin_low = bin_low
        self.bin_high = bin_high
        
        
        # using https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/AnalysisTop/TopDataPreparation/XSection-MC15-13TeV.data
        # for reference
        # x-secs are in pb
        self.xsec_info = {
            "ttbar": 396.87 + 332.97, # nonallhad + allhad, keep same x-sec for all
            "single_top_s_chan": 2.0268 + 1.2676,
            "single_top_t_chan": (36.993 + 22.175)/0.252,  # scale from lepton filter to inclusive
            "single_top_tW": 37.936 + 37.906,
            "wjets": 61457 * 0.252,  # e/mu+nu final states
            "data": None
        }
        print(f'Total number of files: {self.total}')
        

    def _construct_fileset(self):
        n_files_max_per_sample = self.n_files_max_per_sample
        with open ('ntuples.json') as f:
            file_info = json.load(f)
        fileset = {}
        for process in file_info.keys():
            if process == "data":
                continue  # skip data
            fileset[process] = {}
            self[process] = {}
            self._nevts_total[process] = {}
            for variation in file_info[process].keys():
#                 if variation != 'nominal': continue      
                file_list = file_info[process][variation]["files"]
                if n_files_max_per_sample != -1:
                    file_list = file_list[self.first:n_files_max_per_sample]  # use partial set of samples
                file_paths = [f["path"] for f in file_list]
                nevts_total = sum([f["nevts"] for f in file_list])
                self._nevts_total[process].update({variation:nevts_total})
                fileset[process].update({variation: len(file_paths)})
                self.total += len(file_paths)
                self[process][variation] = {}
                
        return fileset

    def fill(self, process, variation, ):
        
        # analysis algorithm themself implemented here
        # fill function accepts parameters pair (process, variation) to which are assigned files in self.input_data
        
        # all operations are handled by RDataFrame class, so the first step is the RDataFrame object instantiating
        input_data = [f"{self.disk}/{process}_{variation}/{i}.root" for i in range(self.nfiles[process][variation])]             
        d = RDataFrame('events', input_data)
        
        # normalization for MC
        x_sec = self.xsec_info[process]
        nevts_total = self._nevts_total[process][variation]
        lumi = 3378 # /pb
        xsec_weight = x_sec * lumi / nevts_total
        d = d.Define('weights', str(xsec_weight)) #default weights
        
        
        
        if variation == 'nominal':
            
            # jet_pt variations definition
            # pt_scale_up() and pt_res_up(jet_pt) return scaling factors applying to jet_pt
            # pt_scale_up() - jet energy scaly systematic
            # pt_res_up(jet_pt) - jet resolution systematic 

            
            d = d.Vary('jet_pt', "ROOT::RVec<ROOT::RVecF>{jet_pt*pt_scale_up(), jet_pt*pt_res_up(jet_pt)}", ["pt_scale_up", "pt_res_up"])
            if process == 'wjets':
                
                # flat weight variation definition
                d = d.Vary('weights', 
                           "weights*flat_variation()",
                           [f"scale_var_{direction}" for direction in ["up", "down"]]
                          )
                
        ### event selection - the core part of the algorithm applied for both regions
        # selecting events containing at least one lepton and four jets with pT > 25 GeV
        # applying requirement at least one of them must be b-tagged jet (see details in the specification)
        d = d.Define('electron_pt_mask', 'electron_pt>25').Define('muon_pt_mask', 'muon_pt>25').Define('jet_pt_mask', 'jet_pt>25')\
             .Filter('Sum(electron_pt_mask) + Sum(muon_pt_mask) == 1')\
             .Filter('Sum(jet_pt_mask) >= 4')\
             .Filter('Sum(jet_btag[jet_pt_mask]>=0.5)>=1')
             
        
        # b-tagging variations for nominal samples
        d = d.Vary('weights', 
                   'ROOT::RVecD{weights*btag_weight_variation(jet_pt[jet_pt_mask])}',
                   [f"{weight_name}_{direction}" for weight_name in [f"btag_var_{i}" for i in range(4)] for direction in ["up", "down"]]
                  ) if variation == 'nominal' else d
        
        

        ## as next steps for different regions are different, there is a fork in the algorithm
        # we create another RDF pointer for each region called "fork"
        measured = {"4j1b": "HT", "4j2b": 'trijet_mass'} # columns names of observables for two regions
        for region in ["4j1b","4j2b"]:
            observable = measured[region]
            
            if region == "4j1b":
                
                # only one b-tagged region required
                # observable is total transvesre momentum 
                fork = d.Filter('Sum(jet_btag[jet_pt_mask]>=0.5)==1').Define(observable, 'Sum(jet_pt[jet_pt_mask])')      

            elif region == "4j2b":
                
                # select events with at least 2 b-tagged jets
                # building four-momentum vectors for each jet
                fork = d.Filter('Sum(jet_btag[jet_pt_mask]>=0.5)>1').Define("jet_p4", 
                    "ROOT::VecOps::Construct<ROOT::Math::PxPyPzMVector>(jet_px[jet_pt_mask], jet_py[jet_pt_mask], jet_pz[jet_pt_mask], jet_mass[jet_pt_mask])"
                )
                
                # building trijet combinations
                fork = fork.Define('trijet', 
                    'ROOT::VecOps::Combinations(jet_pt[jet_pt_mask],3)'
                ).Define('ntrijet', 'trijet[0].size()')

                # assigning four-momentums to each trijet combination
                fork = fork.Define('trijet_p4', 
                                      'ROOT::VecOps::RVec<ROOT::Math::PxPyPzMVector> trijet_p4(ntrijet);'              +\
                                      'for (int i = 0; i < ntrijet; ++i) {'                                            +\
                                          'int j1 = trijet[0][i]; int j2 = trijet[1][i]; int j3 = trijet[2][i];'       +\
                                          'trijet_p4[i] = jet_p4[j1] + jet_p4[j2] + jet_p4[j3];'                       +\
                                      '}'                                                                              +\
                                      'return trijet_p4;'                                                                                                                          
                                     )

                # getting trijet transverse momentum values from four-momentum vectors
                fork = fork.Define('trijet_pt', 
                        'return ROOT::VecOps::Map(trijet_p4, [](ROOT::Math::PxPyPzMVector v) { return v.Pt(); })'
                                            )

              
                # trijet_btag is a helpful array of bool values indicating whether or not the maximum btag value in trijet is larger than 0.5 threshold 
                fork = fork.Define('trijet_btag', 
                                                  'ROOT::VecOps::RVec<bool> btag(ntrijet);'                                   +\
                                                  'for (int i = 0; i < ntrijet; ++i) {'                                       +\
                                                   'int j1 = trijet[0][i]; int j2 = trijet[1][i]; int j3 = trijet[2][i];'     +\
                                                   'btag[i]=std::max({jet_btag[j1], jet_btag[j2], jet_btag[j3]})>0.5;'        +\
                                                  '}'                                                                         +\
                                                  'return btag;'
                                            )
                # find trijet with maximum pt and higher that threshold btag
                # get mass for found jet four-vector 
                # trijet mass themself is an observable quantity
                fork=fork.Define(observable,
                                                  'double mass;'+\
                                                  'double Pt = 0;'+\
                                                  'double indx = 0;'+\
                                                  'for (int i = 0; i < ntrijet; ++i) {'               +\
                                                  '    if ((Pt < trijet_pt[i]) && (trijet_btag[i])) {'+\
                                                  '        Pt = trijet_pt[i];'+\
                                                  '        indx=i;'+\
                                                  '    }'                                            +\
                                                  '}'                                                +\
                                                  'mass = trijet_p4[indx].M();'             +\
                                                  'return mass;'
                                                 )
                
            
            # fill histogram for observable column in RDF object
            res = fork.Histo1D((f'{process}_{variation}_{region}', process, self.num_bins, self.bin_low, self.bin_high), observable, 'weights')
            self.hist.append(res) # save the pointer to further triggering 
            print(f'histogram {region}_{process}_{variation} has been created')
            
            # save pointers for variations
            # self.variations is a temporary container for all pointers
            if variation == 'nominal':
                self.variations[f"{process}__{region}"] = ROOT.RDF.Experimental.VariationsFor(res)
            else:
                self[process][variation][region] = res

    # build 9 Graphs for each data sample            
    def Fill(self):
        self.hist = []
        for process in self:
            
            for variation in self.nfiles[process]:
                self.fill(process=process, variation=variation)

    # run 9 Graphs for each data sample            
    def Accumulate(self):
        ROOT.RDF.RunGraphs(self.hist)  
    
    # transform TtbarAnalysis to dictionary (process, variation, region) -> hitogram
    def TransfToDict(self):
        for key in self.variations.keys():
            hist_map = self.variations[key]
            key = str(key).split('__')
            process = key[0]; region = key[1]
            for hist_name in hist_map.GetKeys():
                variation = 'nominal' if hist_name == 'nominal' else str(hist_name).split(':')[1]
                if variation not in self[process]: self[process][variation] = {}
                hist = hist_map[hist_name]
                if not isinstance(hist, ROOT.TH1D): hist = hist.GetValue()
                analysisManager[process][variation][region] = hist
        analysisManager.ExportJSON()
        
    def GetProcStack(self, region, variation='nominal'):
        return [self[process][variation][region] for process in self]
    
    def GetVarStack(self, region, process="ttbar"):
        return [self[process][variation][region] for variation in self[process]]
    
    # necessary only for sanity checks
    def ExportJSON(self):
        data = {}
        for process in self:
            data[process] = {}
            for variation in self[process]:
                data[process][variation] = [region for region in self[process][variation]]
        with open('data.json', 'w') as f:
            json.dump(data, f)
                
                
                    


disk='/data/ssdext4_agc_data/afalko' if not args.second_disk else '/data/ssdext4_agc_data_2/afalko'
print(f'processing data located at {disk} disk')
analysisManager = TtbarAnalysis(disk=disk)


import time
t0 = time.time()

analysisManager.Fill()
t1 = time.time()
print(f"\npreprocessing took {round(t1 - t0,2)} seconds")
analysisManager.Accumulate()
t2 = time.time()
print(f"processing took {round(t2 - t1,2)} seconds")
print(f"execution took {round(t2 - t0,2)} seconds")


analysisManager.TransfToDict()
analysisManager['ttbar'].keys()



output = ROOT.TFile.Open(FILE, 'RECREATE')
for process in analysisManager:
    for variation in analysisManager[process]:
        for region in analysisManager[process][variation]:
            hist_name = f"{region}_{process}_{variation}" if variation != 'nominal' else f"{region}_{process}"
            hist = analysisManager[process][variation][region]
            if not isinstance(hist, ROOT.TH1D): hist = hist.GetValue() #this this a bag
            if hist.IsZombie(): raise TypeError(hist_name)
            hist_sliced = ROOT.Slice(hist, 120, 550)
            hist_binned = hist_sliced.Rebin(2, hist.GetTitle())
            output.WriteObject(hist_binned, hist_name)
output.Close()


