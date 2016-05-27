# -*- coding: utf-8 -*-

"""
DNN show case config.
"""


import FWCore.ParameterSet.Config as cms


process = cms.Process("dnnShowCase")

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(10))

process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(False))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:/net/scratch_cms/institut_3a/rieger/raw/RunIISpring15DR74_13TeV_25bx_20pu/ttHbb125/file_1.root")
)

process.dnnShowCase = cms.EDAnalyzer("DNNShowCase",
    modelFile     = cms.string("$CMSSW_BASE/src/DNNShowCase/DNNShowCase/data/showcasemodel.pkl"),
    inputName     = cms.string("input"),
    outputName    = cms.string("output"),
    jetCollection = cms.InputTag("slimmedJets")
)

process.p = cms.Path(process.dnnShowCase)
