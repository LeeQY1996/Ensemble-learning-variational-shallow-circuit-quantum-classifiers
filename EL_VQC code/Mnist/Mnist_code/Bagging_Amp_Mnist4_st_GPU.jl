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
dev=3
device!(dev)

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



function train_circuits(nqubit,nlayer,mq,batch_size,train_datas,train_labels,C)
    circuits = Vector{QCircuit}()
    for c in 1:C
        datas,labels=bootstrap(train_datas,train_labels,batch_size)
        statebatch=CuStateVectorBatch(CuArray(datas),nqubit,size(datas)[2])
        circuit = build_circuit(nqubit,nlayer,mq)
        circuit=train_circuit(circuit,statebatch,labels)
        push!(circuits,circuit)
    end
    return circuits
end



function train_circuit(circuit,statebatch,labelbatch)
    opt=ADAM(lr)
    params = parameters(circuit)
    for iter = 1:total_iter
        grad = gradient(x->loss_util(x,statebatch,labelbatch),circuit)[1]
        update!(opt, params,grad)
        reset_parameters!(circuit, params)
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

function bootstrap(datas,labels,Size)
    L,N=size(datas)
    L1,N1=size(labels)
    samplelist=sample(Vector(1:N),Size,replace=true,ordered=true)
    newlist=Vector{Int}()
    for i in samplelist
        if !(i in newlist)
            push!(newlist,i)
        end
    end
    L2=length(newlist)
    newdatas=similar(datas,(L,L2))
    newlabels=similar(labels,(L1,L2))
    for i in 1:L2
        newdatas[:,i]=datas[:,newlist[i]]
        newlabels[:,i]=labels[:,newlist[i]]
    end
    return newdatas,newlabels
end

#分类类别
#分类类别
global const K=4
global nqubit=6
global nlayer=3
batch_size=1541
total_iter=1500
lr=5e-3
mq=[5,6]
const B=[QubitsTerm(mq[1]=>UP),QubitsTerm(mq[2]=>UP)]

train_datas = readdlm("../MNIST_data/Mnist_1357_train_set.csv",',')
train_labels = readdlm("../MNIST_data/Mnist_1357_train_labels.csv",',')
train_set = CuStateVectorBatch(CuArray(train_datas),nqubit,size(train_datas)[2])

test_datas = readdlm("../MNIST_data/Mnist_1357_test_set.csv",',')
test_labels = readdlm("../MNIST_data/Mnist_1357_test_labels.csv",',')
test_set = CuStateVectorBatch(CuArray(test_datas),nqubit,size(test_datas)[2])

Sm=10
# C=10
# Circuits = Vector{Vector{QCircuit}}()
# for i in 1:Sm
#     circuits=train_circuits(nqubit,nlayer,mq,batch_size,train_datas,train_labels,C)
#     push!(Circuits,circuits)
# end
# test_acc=zeros(Sm,C)
# train_acc=zeros(Sm,C)
# for i in 1:Sm
#     for j in 1:C
#     train=fprediction(Circuits[i][1:j],train_set)
#     train_acc[i,j]=dot(train,train_labels)/1541
#     test = fprediction(Circuits[i][1:j],test_set)
#     test_acc[i,j]=dot(test,test_labels)/726
#     end
# end
# save("Bagging_layer$(nlayer)_C$(C)_S$(Sm)_$(dev).jld2","circuits",Circuits,"train_acc",train_acc,"test_acc",test_acc)

# train_acc,test_acc=collect_re(circuitss[1],train_set,test_set,C,train_labels,test_labels)

# println("pre_acc_train:$(mean(train_acc))%;pre_acc_test:$(mean(test_acc))%")
# save("ada_Mnist_4class_23_new.jld2","train_acc",train_acc,"test_acc",test_acc,"circuitss",circuitss,"alphass",alphass)


# rmprocs(PID)