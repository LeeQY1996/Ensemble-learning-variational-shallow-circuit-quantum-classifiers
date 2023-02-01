using CUDA
push!(LOAD_PATH,"../../../package/QuantumCircuits_demo/src","../../../package/VQC_demo_cuda/src")
using VQC
using QuantumCircuits, QuantumCircuits.Gates
using Flux.Optimise:ADAM,update!
using Zygote
using Random
include("ElementOp.jl")
using DelimitedFiles
using Statistics
using StatsBase
using JLD2
using FileIO
device!(3)

function loss_util(circuit,data_batch,label_batch)
    loss_value = 0.
    out=circuit*data_batch
    p01=real.(expectation(B[1],out))
    L=length(p01)
    p02=real.(expectation(B[2],out))
    p1=p01.*p02
    p2=p01.*(-p02.+1)
    p3=(-p01.+1).*p02
    p4=(-p01.+1).*(-p02.+1)
    loss_value -= dot(label_batch[1,:],log.(p1))
    loss_value -= dot(label_batch[2,:],log.(p2))
    loss_value -= dot(label_batch[3,:],log.(p3))
    loss_value -= dot(label_batch[4,:],log.(p4))
    return loss_value/L
end

function fresult(labels::Matrix{Float64},prediction::Matrix{Int64})#判断分类的对错，分类正确：+；分类错误：-
    L=size(labels,2)
    result=zeros(L)
    for i in 1:L
        if labels[:,i]!=prediction[:,i]
            result[i]=1
        end
    end
    return result
end

function fresult(labels::Matrix{Float64},prediction::Matrix{Float64})#判断分类的对错，分类正确：+；分类错误：-
    L=size(labels,2)
    result=zeros(L)
    for i in 1:L
        if labels[:,i]!=prediction[:,i]
            result[i]=1
        end
    end
    return result
end

function fresult(labels::Matrix{Float64},prediction::Matrix{Float64})#判断分类的对错，分类正确：+；分类错误：-
    L=size(labels,2)
    result1=zeros(Int,4,L)
    for i in 1:L
        pos=argmax(prediction[:,i])
        result1[pos,i]=1
    end
    result=zeros(L)
    for i in 1:L
        if labels[:,i]!=result1[:,i]
            result[i]=1
        end
    end
    return result
end

# function Data_Sampling(dataset,labelset,Size)
#     N,L=size(labelset)
#     datalist=sampel(Vector(1:L),Size,replace=true,ordered=false)
#     tmp_dataset=zeros(N,)
#     for i in datalist
#     tmp_dataset=
#     tmp_labelset=
# end

function fprediction_util(circuit::QCircuit,data_set)#输出预测向量.
    L=nitems(data_set)
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
    L=nitems(data_set)
    preds=zeros(4,L)
    for i in 1:length(circuits)
        preds.+=fprediction_util(circuits[i],data_set)
    end
    return preds
end

function fprediction(circuits::Vector{QCircuit},data_set)
    preds=fprediction_util(circuits,data_set)
    L=size(preds,2)
    result=zeros(Int,4,L)
    for i in 1:L
        pos=argmax(preds[:,i])
        result[pos,i]=1
    end
    return result
end



function train_circuits(nqubit,nlayer,mq,C,batch_size,train_datas,train_labels)
    circuits = Vector{QCircuit}()
    statebatchs,labelbatchs=cugenerate_batch(nqubit, batch_size, train_datas,train_labels)
    for c in 1:C
        circuit = build_circuit(nqubit,nlayer,mq)
        circuit=train_circuit(circuit,statebatchs,labelbatchs)
        push!(circuits,circuit)
    end
    return circuits
end



function train_circuit(circuit,statebatchs,labelbatchs)
    opt=ADAM(lr)
    params = parameters(circuit)
    for iter = 1:total_iter
        for (statebatch,labelbatch) in zip(statebatchs,labelbatchs)
            grad = gradient(x->loss_util(x,statebatch,labelbatch),circuit)[1]
            update!(opt, params,grad)
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
#分类类别
global const K=4
global nqubit=6
global nlayer=2
batch_size=256
total_iter=1500
lr=5e-3
mq=[5,6]
const B=[QubitsTerm(mq[1]=>UP),QubitsTerm(mq[2]=>UP)]
C=100

train_datas = readdlm("../MNIST_data/Mnist_1357_train_set.csv",',')
train_labels = readdlm("../MNIST_data/Mnist_1357_train_labels.csv",',')
train_set = CuDensityMatrixBatch(CuArray(todm(train_datas)),nqubit,size(train_datas)[2])

test_datas = readdlm("../MNIST_data/Mnist_1357_test_set.csv",',')
test_labels = readdlm("../MNIST_data/Mnist_1357_test_labels.csv",',')
test_set = CuStateVectorBatch(CuArray(todm(test_datas)),nqubit,size(test_datas)[2])

Sm=1

#circuits = Vector{QCircuit}()
for i in 1:Sm
    circuits=train_circuits(nqubit,nlayer,mq,C,batch_size,train_datas,train_labels)
    save("Random_Forest_3layer_1.jld2","circuits",circuits)
end
# test_acc=zeros(Sm)
# train_acc=zeros(Sm)
# for i in 1:Sm
#     train=fprediction(circuitss[i],train_set)
#     train_acc[i]=dot(train,train_labels)/1541
#     test = fprediction(circuitss[i],test_set)
#     test_acc[i]=dot(test,test_labels)/726
# end


# train_acc,test_acc=collect_re(circuitss[1],train_set,test_set,C,train_labels,test_labels)

# println("pre_acc_train:$(mean(train_acc))%;pre_acc_test:$(mean(test_acc))%")
# save("ada_Mnist_4class_23_new.jld2","train_acc",train_acc,"test_acc",test_acc,"circuitss",circuitss,"alphass",alphass)


# rmprocs(PID)