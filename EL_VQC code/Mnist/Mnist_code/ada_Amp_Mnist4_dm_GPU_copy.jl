using CUDA
push!(LOAD_PATH,"../../../package/QuantumCircuits_demo/src","../../../package/VQC_demo_cuda_32/src")
include("ElementOp.jl")
using VQC
using QuantumCircuits, QuantumCircuits.Gates
using Flux.Optimise:ADAM, update!
using Zygote
using Random
using DelimitedFiles
using Statistics 
using StatsBase
using JLD2
using FileIO
using Dates
dev=0
device!(dev)

function fresult(labels::Matrix{Float64},prediction::Matrix{<:Real})#判断分类的对错，分类正确：+；分类错误：-
    L=size(labels,2)
    result=zeros(L)
    for i in 1:L
        if argmax(labels[:,i])!=argmax(prediction[:,i])
            result[i]=1
        end
    end
    return result
end

function fprediction_util(circuit::QCircuit,data_set)#输出预测向量.
    L=data_set.nitems
    preds=zeros(4,L)
    x=circuit*data_set
    p01=real(expectation(B[1],x)) #测量第一个比特0的概率
    p02=real(expectation(B[2],x)) #测量第二个比特0的概率
    preds[1,:].=p01.*p02
    preds[2,:].=p01.*(-p02.+1)
    preds[3,:].=(-p01.+1).*p02
    preds[4,:].=(-p01.+1).*(-p02.+1)
    return preds
end

function fprediction_util(circuits::Vector{QCircuit},data_set)
    L=length(data_set)
    preds=zeros(4,L)
    for i in 1:length(alphas)
        preds.+=fprediction_util(circuits[i],data_set)
    end
    return preds
end

function fprediction(circuit::QCircuit,data_set)
    preds=fprediction_util(circuit,data_set)
    L=size(preds,2)
    result=zeros(Int,4,L)
    for i in 1:L
        pos=argmax(preds[:,i])
        result[pos,i]=1
    end
    return result
end

function train_circuits(nqubit,nlayer,mq,C,batch_size,train_datas,train_labels,P)
    circuits = Vector{QCircuit}()
    statebatchs,labelbatchs=cugenerate_batch(nqubit, batch_size, train_datas,train_labels,isdm=true)
    for c in 1:C
        circuit = build_circuit(nqubit,nlayer,mq,P)
        circuit = train_circuit(circuit,statebatchs,labelbatchs)
        push!(circuits,circuit)
    end
    return circuits
end

function loss_util(circuit::QCircuit, B::Vector{QubitsTerm}, data_batch::Union{CuDensityMatrixBatch,DensityMatrixBatch}, label_batch::Matrix{<:Real})
    loss_value = 0.
    out=circuit*data_batch
    p01=real.(expectation(B[1],out))
    p02=real.(expectation(B[2],out))
    p1=p01.*p02
    p2=p01.*(-p02.+1)
    p3=(-p01.+1).*p02
    p4=(-p01.+1).*(-p02.+1)
    loss_value -= dot(label_batch[1,:],log.(p1))
    loss_value -= dot(label_batch[2,:],log.(p2))
    loss_value -= dot(label_batch[3,:],log.(p3))
    loss_value -= dot(label_batch[4,:],log.(p4))
    return loss_value/length(p01)
end

function train_circuit(circuit, statebatchs, labelbatchs)
    opt=ADAM(lr)
    params = parameters(circuit)
    for iter = 1:total_iter
        for (statebatch, labelbatch) in zip(statebatchs, labelbatchs)
            grad = gradient(x->loss_util(x, B, statebatch, labelbatch), circuit)[1]
            update!(opt, params, grad)
            reset_parameters!(circuit, params)
        end
    end
    return circuit
end

function collect_re(circuits,train_set,test_set,C,train_labels,test_labels)
    train_acc=zeros(C)
    test_acc=zeros(C)
    train_acc_tmp=zeros(4,1541)
    test_acc_tmp=zeros(4,726)
    for j in 1:C
        train_acc_tmp+=fprediction_util(circuits[j],train_set)
        re1=fresult(train_labels,train_acc_tmp)
        train_acc[j]=1-sum(re1)/1541
        test_acc_tmp+=fprediction_util(circuits[j],test_set)
        re2=fresult(test_labels,test_acc_tmp)
        test_acc[j]=1-sum(re2)/726
    end
    return train_acc,test_acc
end

#分类类别
global const K = 4 
global nqubit = 6
global nlayer = 12
batch_size = 1536
total_iter = 1000
lr = 5e-3
mq = [5,6]
const B = [QubitsTerm(mq[1]=>UP), QubitsTerm(mq[2]=>UP)]
C = 20

train_datas = readdlm("../MNIST_data/Mnist_1357_train_set.csv",',')
train_labels = readdlm("../MNIST_data/Mnist_1357_train_labels.csv",',')
train_set = CuDensityMatrixBatch(CuArray(todm(train_datas)), nqubit, size(train_datas)[2])


test_datas = readdlm("../MNIST_data/Mnist_1357_test_set.csv",',')
test_labels = readdlm("../MNIST_data/Mnist_1357_test_labels.csv",',')
test_set  = CuDensityMatrixBatch(CuArray(todm(test_datas)), nqubit, size(test_datas)[2])

Sm=1
println("开始训练")
Ps=[0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12,0.14,0.16,0.18]

for j in dev*3+1:(dev+1)*3
    P=Ps[j]
    for i in 1:Sm
        circuits=train_circuits(nqubit, nlayer, mq, C, batch_size, train_datas, train_labels, P)
        f=jldopen("deep_amp_data/$(P)_ada3_amp_dm_cu_c&a.jld2","a+")
        f["circuits$(i)"] = circuits
        f["alphas$(i)"] = alphas
        close(f)
        train_acc,test_acc=collect_re(circuits,train_set,test_set,C,train_labels,test_labels)
        open("deep_amp_data/$(P)_ada3_amplitude_cu_acc_c$(C)s$(Sm).csv","a+") do io
            writedlm(io,[train_acc,test_acc])
        end
        open("deep_amp_data/$(P)_process.txt","a+") do io
            writedlm(io,["$(P):训练完成第$(i)个,剩余$(Sm-i).完成时间：$(now())"])
        end
        println("$(P):训练完成第$(i)个,剩余$(Sm-i).完成时间：$(now())")
    end
end