push!(LOAD_PATH,"../../../QuantumCircuits/src","../../../VQC.jl/src")
using VQC
using QuantumCircuits, QuantumCircuits.Gates
using Random
using Flux.Optimise:ADAM,update!
using Statistics
using StatsBase

#label:0 or 1 

function load_Qdata(datas,labels)
    L,N=size(datas)
    data_set=[[StateVector(datas[i,:]),labels[i],1/L] for i in 1:L]
    return data_set
end

function load_Qdata(datas,labels)
    L,N=size(datas)
    data_set=[[StateVector(datas[i,:]),labels[i],1/L] for i in 1:L]
    return data_set
end

function build_circuit_xy(N,nlayer,mq)
    circuit = QCircuit()
    for k in 1:nlayer
        for j=1:N
            push!(circuit, RxGate(j, rand(), isparas=true))
            push!(circuit, RyGate(j, rand(), isparas=true))
            push!(circuit, RxGate(j, rand(), isparas=true))
        end 
        for j=1:N-1
            push!(circuit,CNOTGate(j,j+1))
        end
        push!(circuit,CNOTGate(N,1))
    end
    push!(circuit, RxGate(mq[1], rand(), isparas=true))
    push!(circuit, RyGate(mq[1], rand(), isparas=true))
    push!(circuit, RxGate(mq[1], rand(), isparas=true))
    return circuit
end


function build_circuit_zy(N,nlayer)
    circuit = QCircuit()
    for k in 1:nlayer
        for j=1:N
            push!(circuit, RzGate(j, rand(), isparas=true))
            push!(circuit, RyGate(j, rand(), isparas=true))
            push!(circuit, RzGate(j, rand(), isparas=true))
        end 
        for j=1:N-1
            push!(circuit,CNOTGate(j,j+1))
        end
        push!(circuit,CNOTGate(N,1))
    end
    push!(circuit, RzGate(mq[1], rand(), isparas=true))
    push!(circuit, RyGate(mq[1], rand(), isparas=true))
    push!(circuit, RzGate(mq[1], rand(), isparas=true))
    return circuit
end

function generate_batch(train_set,batch_size)
    L=length(train_set)
    tmp=shuffle(train_set)
    pos=1
    data_batchs=[]
    while pos < L-batch_size
        push!(data_batchs,tmp[pos:pos+batch_size-1])
        pos+=batch_size
    end 
    push!(data_batchs,tmp[pos:end])
    return data_batchs
end


function fresult(labels::Vector{Vector{Float64}},prediction::Matrix{Int})#判断分类的对错，分类正确：+；分类错误：-
    L=size(labels,1)
    result=zeros(L)
    for i in 1:L
        if labels[i]!=prediction[i,:]
            result[i]=1
        end
    end
    return result
end

function fprediction(circuits::Vector{QCircuit},alphas::Vector{Float64},data_set)
    preds=fprediction_util(circuits,alphas,data_set)
    L=size(preds,1)
    result=zeros(Int,L,4)
    for i in 1:L
        pos=argmax(preds[i,:])
        result[i,pos]=1
    end
    return result
end


function fprediction_util(circuit::QCircuit,data_set)#输出预测向量.
    L=length(data_set)
    preds=zeros(L)
    for i in 1:L
        x=circuit*data_set[i][1]
        p=real(expectation(B[1],x))[1] #测量第一个比特0的概率
        p=(p+1)/2
        preds[i]=round(p)
    end
    return preds
end

function fprediction_util(circuits::Vector{QCircuit},alphas::Vector{Float64},data_set)
    L=length(data_set)
    preds=zeros(L,K)
    for i in 1:length(alphas)
        preds.+=fprediction_util(circuits[i],data_set).*alphas[i]
    end
    return preds
end

function fprediction(circuits::Vector{QCircuit},alphas::Vector{Float64},data_set)
    preds=fprediction_util(circuits,alphas,data_set)
    L=size(preds,1)
    result=zeros(Int,L,K)
    for i in 1:L
        pos=argmax(preds[i,:])
        result[i,pos]=1
    end
    return result
end
