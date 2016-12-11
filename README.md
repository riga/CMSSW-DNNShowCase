## CMSSW-DNNShowCase

Proof-of-concept showing how DNNs can be used within CMSSW using
[tfdeploy](https://github.com/riga/tfdeploy).


#### Installation and execution

```shell
# setup CMSSW
source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsrel CMSSW_8_0_24
cd CMSSW_8_0_24/src

# add the show case and compile
git clone https://github.com/riga/CMSSW-DNNShowCase DNNShowCase
scram b
cmsenv

# start the test
cmsRun DNNShowCase/DNNShowCase/test/showcase_cfg.py
```
