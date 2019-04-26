from esrf_read import *;

test_scan_xml="/diskstation/data/xns/maxibone/esrf_dental_implants_2013/810c-repeat/HA_xc520_50kev_1_88mu_implant_810c_001repeat_pag.xml";
info = esrf_read_xml(test_scan_xml);

region = [[1000,1000,0],[1300,1300,798]]
