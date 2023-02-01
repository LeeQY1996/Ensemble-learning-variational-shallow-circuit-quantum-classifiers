push!(LOAD_PATH,"../../package/QuantumCircuits_demo_32/src","../../package/VQC_demo_cuda/src")
using VQC
using QuantumCircuits, QuantumCircuits.Gates
using DelimitedFiles
using Random
using Flux.Optimise:ADAM,update!
using Base.Threads
using Statistics
using StatsBase
using Zygote
using Zygote: @adjoint
using Base: @kwdef
using Plots
using CUDA
using JLD2
device!(2)

function build_circuit_zx(N,nlayer)
    circuit = QCircuit()
    for k in 1:nlayer
        for j=1:N
            push!(circuit, RxGate(j, rand(), isparas=true))
            push!(circuit, RzGate(j, rand(), isparas=true))
            push!(circuit, RxGate(j, rand(), isparas=true))
        end 
        for j=1:N-1
            push!(circuit,CNOTGate(j,j+1))
        end
    end
    push!(circuit, RxGate(mq[1], rand(), isparas=true))
    push!(circuit, RzGate(mq[1], rand(), isparas=true))
    push!(circuit, RxGate(mq[1], rand(), isparas=true))
    return circuit
end


function loss_util(circuit,data_batch,label_batch)
    loss_value = 0.
    out=circuit*data_batch
    p01=real.(expectation(B[1],out))
    L=length(p01)
    p1=p01
    p2=-p01.+1
    loss_value -= dot(label_batch[1,:],log.(p1))
    loss_value -= dot(label_batch[2,:],log.(p2))
    return loss_value/L
end


function fprediction_util(circuit::QCircuit,data_set)#输出预测向量.
    L=nitems(data_set)
    preds=zeros(2,L)
    x=circuit*data_set
    p=real(expectation(B[1],x)) #测量第一个比特0的概率
    preds[1,:]=p
    preds[2,:]=-p.+1
    return preds
end

function fprediction_util(circuits::Vector{QCircuit},data_set)#输出预测向量.
    L=nitems(data_set)
    preds=zeros(2,L)
    for circuit in circuits
        x=circuit*data_set
        p=real(expectation(B[1],x)) #测量第一个比特0的概率
        preds[1,:].+=p
        preds[2,:].+=(-p.+1)
    end
    return preds
end

function fprediction(circuits::Vector{QCircuit},data_set)#输出预测向量.
    L=nitems(data_set)
    preds=fprediction_util(circuits::Vector{QCircuit},data_set)./length(circuits)
    for i in 1:L
        preds[1,i]>preds[2,i] && (preds[:,i]=[1.,0.])
        preds[1,i]<preds[2,i] && (preds[:,i]=[0.,1.])
    end
    return preds
end

@kwdef mutable struct Args
    η::Float64 = 5e-3       ## learning rate
    batchsize::Int = 40    ## batch size
    epochs::Int = 1000      ## number of epochs
end
global args = Args(;)
#分类类别
global const K=2
global N=15
global nlayer=2
global mq=[N] #measure_qubits
#测量算子
UP=[1 0; 0 0]
DOWN=[0 0; 0 1]
const B=[QubitsTerm(mq[1]=>UP),QubitsTerm(mq[1]=>"X"),QubitsTerm(mq[1]=>DOWN)]
# Samples=200
# datas = readdlm("SPT_train_set_$(N)_$(Samples)_0.05.csv",',', Float64)
# labels = readdlm("SPT_train_label_$(N)_$(Samples)_0.05.csv",',')
datas1 = readdlm("SPT_train_set_$(N)_$(400)_0.01.csv",',', Float64)
labels1 = readdlm("SPT_train_label_$(N)_$(400)_0.01.csv",',')
datas2 = readdlm("SPT_train_set_$(N)_$(40).csv",',', Float64)
labels2 = readdlm("SPT_train_label_$(9)_$(40).csv",',')
datas=zeros(2^N,440)
datas[:,1:400]=datas1
datas[:,401:440]=datas2
labels=zeros(2,440)
labels[:,1:400]=labels1
labels[:,401:440]=labels2

Samples=440
dm=CuStateVectorBatch(CuArray(datas),N,Samples)
C=10

# circuits = Vector{QCircuit}()
# for c in 1:C
#     circuit = build_circuit_zx(N,nlayer)
#     opt=ADAM(args.η)
#     params = parameters(circuit)
#     for iter = 1:args.epochs
#         grad = gradient(x->loss_util(x,dm,labels),circuit)[1]
#         update!(opt, params,grad)
#         reset_parameters!(circuit, params)
#     end
#     push!(circuits,circuit)

#     #测试当前线路集性能
#     x1=fprediction(circuits,dm)
#     re1=dot(x1,labels)
#     x2=fprediction([circuit],dm)
#     re2=dot(x2,labels)
#     println("$(c)th classifier_acc:$(sum(re2)/Samples); the random forest acc:$(sum(re1)/Samples)")
# end  

#test_set = readdlm("SPT_test_set_$(N).csv",',', Float64)

h1_vals = [0.1000, 0.2556, 0.4111, 0.5667, 0.7222, 0.8778, 1.0333, 1.1889, 1.3444, 1.5000]
anti_ferro_mag_boundary = [-1.004, -1.0009, -1.024, -1.049, -1.079, -1.109, -1.154,  -1.225, -1.285, -1.35]
para_mag_boundary = [0.8439, 0.6636, 0.5033, 0.3631, 0.2229, 0.09766, -0.02755, -0.1377, -0.2479, -0.3531]

#circuits=load("adaboost_$(Samples)_$(N)q_$(nlayer)l_100.jld2")["circuits"]
res_pix=zeros(64,64)
res = zeros(64,64)
x=7
for i in 1:64
    test_dm = CuStateVectorBatch(CuArray(test_set[:,(i-1)*64+1:i*64]),N,64)
    re=fprediction(circuits[1:x],test_dm);
    res[:,i]=re[1,:]
end
res[33,2]=0
heatmap(LinRange(0,1.6,64)[2:64],LinRange(-1.6,1.6,64),res[:,2:64],c=cgrad(:summer,rev=true))
plot!(h1_vals,[anti_ferro_mag_boundary para_mag_boundary],markershapes=:diamond,markersize=5,lw=3,linestyle=:dash,lc=["red" "blue"],mc=:orange,fc="white",legend=false,tickfontsize=12,dpi=150)
annotate!([0.75 0.75 0.75 0.53] ,[1 -1.45 -0.5 1.46],[text("Paramagnetic",16), text("Antiferromagnetic",16), text("SPT",16),text("AdaBoost \$D_l\$=$(2) \$L_c\$=$(x)",18)])
savefig("ada_$(x).png")
# save("random_forest_$(Samples)_$(N)q_$(nlayer)l_300.jld2","circuits",circuits,"res",res)

# circuitss=[]
# for i in 1:4
# push!(circuitss,load("random_forest_$(Samples)_$(N)q_$(nlayer)l_$(i)0.jld2")["circuits"])
# end
# for i in 1:64
#     test_dm = CuStateVectorBatch(CuArray(test_set[:,(i-1)*64+1:i*64]),N,64)
#     for circuits in circuitss
#         re=fprediction(circuits[1:5],test_dm);
#         res[:,i]+=re[1,:]
#     end
# end
# res.=res./4
# for i in 1:64
#     for j in 1:64
#         if res[i,j]>0.5
#             res[i,j]=1
#         else
#             res[i,j]=0
#         end
#     end
# end

# heatmap(LinRange(0,1.6,64),LinRange(-1.6,1.6,64),res,c=cgrad([:yellow,:green]))
# plot!(h1_vals,[anti_ferro_mag_boundary para_mag_boundary],markershapes=:diamond,markersize=5,lw=3,linestyle=:dash,lc=["red" "blue"],mc=:orange,fc="white",legend=false,title="random forest_$(Samples)")
# annotate!([0.75 0.75 0.75] ,[1 -1.5 -0.5],["Paramagnetic","Antiferromagnetic", "SPT"])