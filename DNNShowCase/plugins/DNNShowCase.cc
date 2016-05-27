// DNN show case that demonstrates the usage of tfdeploy within a CMSSW module.
// The model has no deeper meaning, it's rather a technical proof-of-concept.

#include <memory>
#include <iostream>
#include "Python.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"


// python evaluation script
static const std::string __script = "\
import os, numpy as np, tfdeploy as td\n\
inp, outp = None, None\n\
def setup(model_file, inp_name, outp_name):\n\
  global inp, outp\n\
  model = td.Model(os.path.expandvars(os.path.expanduser(model_file)))\n\
  inp = model.get(inp_name)\n\
  outp = model.get(outp_name)\n\
def eval(*values):\n\
  return outp.eval({inp: np.array(values).astype(np.float32)})\n\
";


class DNNShowCase: public edm::one::EDAnalyzer<edm::one::SharedResources>
{
public:
  explicit DNNShowCase(const edm::ParameterSet&);
  ~DNNShowCase();

  // parameters
  std::string _modelFile;
  std::string _inputName;
  std::string _outputName;
  edm::InputTag _jetCollection;

  // python objects
  PyObject* _pyContext;
  PyObject* _pyEval;
  PyObject* _pyEvalArgs;

  // tokens
  edm::EDGetTokenT<std::vector<pat::Jet> > _jetToken;

private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;

  double evalModel(const pat::Jet&);
};

DNNShowCase::DNNShowCase(const edm::ParameterSet& iConfig)
{
  usesResource("TFileService");

  // get parameters
  _modelFile     = iConfig.getParameter<std::string>("modelFile");
  _inputName     = iConfig.getParameter<std::string>("inputName");
  _outputName    = iConfig.getParameter<std::string>("outputName");
  _jetCollection = iConfig.getParameter<edm::InputTag>("jetCollection");

  // initialize python
  Py_Initialize();
  PyEval_InitThreads();

  // run the __script
  PyObject* pyMainModule = PyImport_AddModule("__main__");
  PyObject *pyMainDict = PyModule_GetDict(pyMainModule);
  _pyContext = PyDict_Copy(pyMainDict);
  PyRun_String(__script.c_str(), Py_file_input, _pyContext, _pyContext);

  // get the eval function
  _pyEval = PyDict_GetItemString(_pyContext, "eval");
  _pyEvalArgs = PyTuple_New(3); // the model has an input-dimension of 3

  // setup the tfdeploy model
  PyObject* pySetup = PyDict_GetItemString(_pyContext, "setup");
  PyObject* pyArgs = PyTuple_New(3);
  PyTuple_SetItem(pyArgs, 0, PyString_FromString(_modelFile.c_str()));
  PyTuple_SetItem(pyArgs, 1, PyString_FromString(_inputName.c_str()));
  PyTuple_SetItem(pyArgs, 2, PyString_FromString(_outputName.c_str()));
  PyObject* pyResult = PyObject_CallObject(pySetup, pyArgs);
  if (pyResult == NULL) {
    if (PyErr_Occurred() != NULL) {
      PyErr_PrintEx(0);
    }
    throw std::runtime_error("An error occured while loading the tfdeploy model");
  }

  // setup tokens
  _jetToken = consumes<std::vector<pat::Jet> >(_jetCollection);
}


DNNShowCase::~DNNShowCase()
{
  // cleanup python objects
  Py_DECREF(_pyEvalArgs);
  Py_DECREF(_pyEval);
  Py_DECREF(_pyContext);
  Py_Finalize();
}


void DNNShowCase::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // get jets
  edm::Handle<std::vector<pat::Jet> > jets;
  iEvent.getByToken(_jetToken, jets);

  // call the eval model per jet
  // the returned double could also be added to the user floats
  for (const auto& jet: *jets) {
    std::cout << evalModel(jet) << std::endl;
  }
}


double DNNShowCase::evalModel(const pat::Jet& jet)
{
  // set eval arguments: 0 -> pt, 1 -> abs(eta), 2 -> nConstituents
  PyTuple_SetItem(_pyEvalArgs, 0, PyFloat_FromDouble(jet.pt()));
  PyTuple_SetItem(_pyEvalArgs, 1, PyFloat_FromDouble(abs(jet.eta())));
  PyTuple_SetItem(_pyEvalArgs, 2, PyFloat_FromDouble(jet.nConstituents()));

  // evaluation
  return PyFloat_AsDouble(PyObject_CallObject(_pyEval, _pyEvalArgs));
}


void DNNShowCase::beginJob()
{
}


void DNNShowCase::endJob() 
{
}


DEFINE_FWK_MODULE(DNNShowCase);
