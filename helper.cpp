#include "TH1D.h"
#include <string>
#include "TRandom2.h"
#include "ROOT/RVec.hxx"


TH1D SliceHisto (TH1D h, int xfirst, int xlast)
{

   // do slice in xfirst:xlast including xfirst and xlast                                                                           
   TH1D res((std::string("h_sliced_")+h.GetTitle()).c_str(), h.GetTitle(),xlast-xfirst, h.GetXaxis()->GetBinLowEdge(xfirst), h.GetXaxis()->GetBinUpEdge(xlast-1));
   // note that histogram arrays are : [ undeflow, bin1, bin2,....., binN, overflow]                                                                             
   std::copy(h.GetArray()+xfirst,h.GetArray()+xlast,res.GetArray()+1);
   // set correct underflow/overflows
   res.SetBinContent(0, h.Integral(0,xfirst-1));   // set underflow value                                                            
   res.SetBinContent(res.GetNbinsX()+1,h.Integral(xlast,h.GetNbinsX()+1)); // set overflow value                                    
    
  
    return res;
}

TH1D Slice(TH1D h, double low_edge, double high_edge)
{
    int xfirst = h.FindBin(low_edge);
    int xlast = h.FindBin(high_edge);
    return SliceHisto(h, xfirst, xlast);
}

ROOT::VecOps::RVec<float> pt_res_up(const ROOT::VecOps::RVec<float>& jet_pt)
{
    ROOT::VecOps::RVec<float> res(jet_pt.size());
    TRandom2 rnmd;
    for (auto& e: res) {rnmd.SetSeed(0); e = rnmd.Gaus(1,0.05);}
    return res;
}

float pt_scale_up(){
    return 1.03;
}


ROOT::VecOps::RVec<float> btag_weight_variation (const ROOT::VecOps::RVec<float>& jet_pt) {
   ROOT::VecOps::RVec<float> res;
   for (const float& pt: ROOT::VecOps::Take(jet_pt,4)) {
       res.push_back(1+.075*pt/50); res.push_back(1-.075*pt/50);
   }
   return res;
}



ROOT::VecOps::RVec<float> flat_variation(){
    return 1 + ROOT::VecOps::RVec<float>({.025, -.025});
}


