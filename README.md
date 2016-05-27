## CMSSW-DNNShowCase

Proof-of-concept showing how DNNs can be used within CMSSW using
[tfdeploy](https://github.com/riga/tfdeploy).


#### Installation and execution

```shell
# setup CMSSW
source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsrel CMSSW_8_0_8
cd CMSSW_8_0_8/src

# add the show case and compile
git clone https://github.com/riga/CMSSW-DNNShowCase DNNShowCase
scram b
cmsenv

# adjust the python path so that tfdeploy can be imported
export PYTHONPATH="$PYTHONPATH:$CMSSW_BASE/src/DNNShowCase/DNNShowCase/test"

# start the test
cmsRun $CMSSW_BASE/src/DNNShowCase/DNNShowCase/test/showcase_cfg.py
```
