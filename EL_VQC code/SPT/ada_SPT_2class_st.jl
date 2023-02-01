push!(LOAD_PATH,"../../package/QuantumCircuits_demo/src","../../package/VQC_demo_cuda/src")
using VQC
using QuantumCircuits, QuantumCircuits.Gates
using DelimitedFiles
using Random
using Flux.Optimise:ADAM,update!
using Statistics
using StatsBase
using Zygote
using Zygote: @adjoint
using Base: @kwdef
using Plots
using CUDA
using JLD2
device!(1)

function build_circuit_xz(N,nlayer)
    circuit = QCircuit()
    for k in 1:nlayer
        for j=1:N
            push!(circuit, RxGate(j, rand(), isparas=true))
            push!(circuit, RzGate(j, rand(), isparas=true))
            push!(circuit, RxGate(j, rand(), isparas=true))
        end 
        for j=1:N-1#div(N,2)
            push!(circuit,CNOTGate(j,j+1))
            #push!(circuit,CNOTGate(N+1-j,N-j) )
        end
    end
    push!(circuit, RxGate(mq[1], rand(), isparas=true))
    push!(circuit, RzGate(mq[1], rand(), isparas=true))
    push!(circuit, RxGate(mq[1], rand(), isparas=true))
    return circuit
end


function loss_util(circuit,data_batch,label_batch,w)
    loss_value = 0.
    out=circuit*data_batch
    p01=real.(expectation(B[1],out))
    L=length(p01)
    p1=p01
    p2=-p01.+1
    loss_value -= dot(label_batch[1,:].*w,log.(p1))
    loss_value -= dot(label_batch[2,:].*w,log.(p2))
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

function fprediction_util(circuits::Vector{QCircuit},alphas::Vector,data_set)#输出预测向量.
    L=nitems(data_set)
    preds=zeros(2,L)
    for i in 1:length(circuits)
        x=circuits[i]*data_set
        p=real(expectation(B[1],x)) #测量第一个比特0的概率
        preds[1,:].+=(p.*alphas[i])
        preds[2,:].+=((-p.+1).*alphas[i])
    end
    return preds
end

function fprediction(circuits::Vector{QCircuit},alphas::Vector,data_set)#输出预测向量.
    L=nitems(data_set)
    preds=fprediction_util(circuits,alphas,data_set)./length(circuits)
    for i in 1:L
        preds[1,i]>preds[2,i]&& (preds[:,i]=[1.,0.])
        preds[1,i]<preds[2,i] && (preds[:,i]=[0.,1.])
    end
    return preds
end
function fresult(labels::Matrix,prediction::Matrix)#判断分类的对错，分类正确：+；分类错误：-
    L=size(labels,2)
    result=zeros(L)
    for i in 1:L
        if labels[:,i]!=prediction[:,i]
            result[i]=1
        end
    end
    return result
end

@kwdef mutable struct Args
    η::Float64 = 1e-2       ## learning rate
    batchsize::Int = 40    ## batch size
    epochs::Int =1000     ## number of epochs
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
state=CuStateVectorBatch(CuArray(datas),N,Samples)
C=10
# circuits = Vector{QCircuit}()
# alphas = Float64[]
# global W=ones(Samples)./Samples
# for c in 1:C
#     circuit = build_circuit_xz(N,nlayer)
#     opt=ADAM(args.η)
#     params = parameters(circuit)
#     for iter = 1:args.epochs
#         grad = gradient(x->loss_util(x,state,labels,W),circuit)[1]
#         update!(opt, params,grad)
#         reset_parameters!(circuit, params)
#     end
#     #第c个弱分类其训练完成
#     pred=fprediction([circuit],[1.],state)
#     result=fresult(labels,pred)
#     ferror=result'*W
#     if ferror == 0
#         alpha = 10
#         push!(alphas,alpha)
#         push!(circuits,circuit)
#         break
#     else
#         alpha=log((1-ferror)/ferror)
#     end
#     println("error rate:$(ferror)")

#     #更新权重
#     W.=W.*exp.(alpha*result)
#     W./=sum(W)
#     push!(alphas,alpha)
#     push!(circuits,circuit)

#     #测试当前线路集性能
#     x1=fprediction(circuits,alphas,state)
#     re1=dot(x1,labels)
#     println("pre_acc_train:$(sum(re1)/Samples)")
#     if 1-sum(re1)/Samples<0.05
#         break
#     end
# end

circuits=load("adaboost_$(Samples)_$(N)q_$(nlayer)l_200.jld2")["circuits"];
alphas=load("adaboost_$(Samples)_$(N)q_$(nlayer)l_200.jld2")["alphas"];
h1_vals = [0.1000, 0.2556, 0.4111, 0.5667, 0.7222, 0.8778, 1.0333, 1.1889, 1.3444, 1.5000];
anti_ferro_mag_boundary = [-1.004, -1.0009, -1.024, -1.049, -1.079, -1.109, -1.154,  -1.225, -1.285, -1.35];
para_mag_boundary = [0.8439, 0.6636, 0.5033, 0.3631, 0.2229, 0.09766, -0.02755, -0.1377, -0.2479, -0.3531];

#test_set = readdlm("SPT_test_set_$(N).csv",',', Float64);
res = zeros(64,64);
for i in 1:64
    test_dm = CuStateVectorBatch(CuArray(test_set[:,(i-1)*64+1:i*64]),N,64)
    re=fprediction_util(circuits[1:8],alphas,test_dm);
    res[:,i]=re[1,:]
end

# # save("adaboost_$(Samples)_$(N)q_$(nlayer)l_200.jld2","circuits",circuits,"alphas",alphas,"res",res)
res[33,2]=0

heatmap(LinRange(0,1.6,64)[2:64],LinRange(-1.6,1.6,64),res[:,2:64],c=cgrad([:yellow,:green]))
plot!(h1_vals,[anti_ferro_mag_boundary para_mag_boundary],markershapes=:diamond,markersize=5,lw=3,linestyle=:dash,lc=["red" "blue"],mc=:orange,fc="white",legend=false,tickfontsize=16,dpi=150)
annotate!([0.75 0.75 0.75] ,[1 -1.45 -0.5],["Paramagnetic","Antiferromagnetic", "SPT"])