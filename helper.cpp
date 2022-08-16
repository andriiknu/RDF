#include "TH1D.h"


TH1D SliceHisto (TH1D h, int xfirst, int xlast)
{

   // do slice in xfirst:xlast including xfirst and xlast                                                                           
   TH1D res("res","slice of h",xlast-xfirst, h.GetXaxis()->GetBinLowEdge(xfirst), h.GetXaxis()->GetBinUpEdge(xlast-1));
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

