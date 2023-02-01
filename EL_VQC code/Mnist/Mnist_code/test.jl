# #using CUDA
# push!(LOAD_PATH,"../../../package/QuantumCircuits_demo/src","../../../package/VQC_demo_cuda/src")
# using VQC
# using QuantumCircuits, QuantumCircuits.Gates
# using Flux.Optimise:ADAM,update!
# using Zygote
# using Random
# include("ElementOp.jl")
# using DelimitedFiles
# using Statistics 
# using StatsBase
# using JLD2
# using FileIO
# using Dates

P=[0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12,0.14,0.16,0.18]
test_acc=zeros(10,10,12)
for i in 1:length(P)
    acc=readdlm("Ada_Amp_data/new_ada_2/$(P[i])
    _ada3_amplitude_cu_acc_c10s10.csv")
    test_acc[:,:,i]=acc
end
acc= readdlm("Ada_Amp_data/new_ada_2/0.01_ada3_amplitude_cu_acc_c10s10.csv")