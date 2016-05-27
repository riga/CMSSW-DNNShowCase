# -*- coding: utf-8 -*-

"""
DNN show case config.
"""


import FWCore.ParameterSet.Config as cms


process = cms.Process("dnnShowCase")

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(10))

process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(False))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIIFall15MiniAODv2/ttHTobb_M125_13TeV_powheg_pythia8/MINIAODSIM/PU25nsData2015v1_76X_mcRun2_asymptotic_v12-v1/50000/504EAC07-B8B8-E511-A75C-00259029E7FC.root")
)

process.dnnShowCase = cms.EDAnalyzer("DNNShowCase",
    modelFile     = cms.string("$CMSSW_BASE/src/DNNShowCase/DNNShowCase/data/showcasemodel.pkl"),
    inputName     = cms.string("input"),
    outputName    = cms.string("output"),
    jetCollection = cms.InputTag("slimmedJets")
)

process.p = cms.Path(process.dnnShowCase)
