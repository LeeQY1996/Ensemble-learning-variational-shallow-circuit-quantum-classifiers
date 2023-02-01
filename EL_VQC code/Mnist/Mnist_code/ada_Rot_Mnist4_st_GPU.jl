push!(LOAD_PATH,"../../../package/QuantumCircuits_demo/src","../../../package/VQC_demo_cuda/src")
using CUDA
using VQC
using QuantumCircuits, QuantumCircuits.Gates
using Flux.Optimise:ADAM,update!
using Zygote
using Zygote: @adjoint
using Random
using DelimitedFiles
include("ElementOp.jl")
using Statistics
using StatsBase
using JLD2
using FileIO
dev=3
device!(dev)

function build_circuit(nqubit,nlayer,mq,batchsize)
    circuit = QCircuit()
    for k in 1:nlayer
        if k<9
            for j=1:nqubit
                push!(circuit, ERyGate(j, rand(batchsize), batchsize))
            end
        end
        for j=1:nqubit
            push!(circuit, RxGate(j, rand(), isparas=true))
            push!(circuit, RzGate(j, rand(), isparas=true))
            push!(circuit, RxGate(j, rand(), isparas=true))
        end 
        for j=1:nqubit-1
            push!(circuit,CNOTGate(j,j+1))
        end
    end
    for j in mq
    push!(circuit, RxGate(j, rand(), isparas=true))
    push!(circuit, RzGate(j, rand(), isparas=true))
    push!(circuit, RxGate(j, rand(), isparas=true))
    end
    return circuit
end

function set_data!(circuit::QCircuit,data::Matrix{Float64})
    pos=1
    b=size(data,2)
    for gate in circuit
        if typeof(gate)==ERyGate
            gate.paras=data[pos,:]
            gate.batch=(b,)
            pos+=1
        end
    end
end

@adjoint set_data!(circuit,data)=set_data!(circuit,data), z->(z,nothing)

function loss_util(circuit::QCircuit,B::Vector{QubitsTerm},data_batch::Matrix{<:Real},label_batch::Matrix{<:Real})
    loss_value = 0.
    b=size(data_batch,2)
    set_data!(circuit,data_batch)
    tmp=circuit*ini_custate
    p01=real.(expectation(B[1],tmp))
    p02=real.(expectation(B[2],tmp))
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


function generate_batch(batch_size, dataset, labels ; isweight::Bool=false)
    L,N=size(dataset)
    data_batchs=Vector{Matrix}()
    label_batchs=Vector{Matrix{Float64}}()
    pos=1
    while pos <= N-batch_size+1
        push!(data_batchs,dataset[:,pos:pos+batch_size-1]) 
        push!(label_batchs,labels[:,pos:pos+batch_size-1])
        pos+=batch_size
    end
    # push!(data_batchs,dataset[:,pos:end])
    # push!(label_batchs,labels[:,pos:end])
    #把权重和label合并了
    if isweight
        for i in label_batchs
            i.=i./pos
        end
    end
    return data_batchs,label_batchs
end


function fresult(labels::Matrix{Int64},prediction::Matrix{Int64})#判断分类的对错，分类正确：+；分类错误：-
    L=size(labels,2)
    result=zeros(L)
    for i in 1:L
        if labels[:,i]!=prediction[:,i]
            result[i]=1
        end
    end
    return result
end


function fresult(labels::Matrix{Int64},prediction::Matrix{Float64})#判断分类的对错，分类正确：+；分类错误：-
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
    L=size(data_set,2)
    preds=zeros(4,L)
    set_data!(circuit,data_set)
    state=CuStateVectorBatch(nqubit,L)
    x=circuit*state
    p01=real(expectation(B[1],x)) #测量第一个比特0的概率
    p02=real(expectation(B[2],x)) #测量第二个比特0的概率
    preds[1,:].=p01.*p02
    preds[2,:].=p01.*(-p02.+1)
    preds[3,:].=(-p01.+1).*p02
    preds[4,:].=(-p01.+1).*(-p02.+1)
    return preds
end

function fprediction_util(circuits::Vector{QCircuit},alphas::Vector{Float64},data_set)
    L=length(data_set)
    preds=zeros(4,L)
    for i in 1:length(alphas)
        preds.+=fprediction_util(circuits[i],data_set).*alphas[i]
    end
    return preds
end

function fprediction(circuits::Vector{QCircuit},alphas::Vector{Float64},data_set)
    preds=fprediction_util(circuits,alphas,data_set)
    L=size(preds,2)
    result=zeros(Int,4,L)
    for i in 1:L
        pos=argmax(preds[:,i])
        result[pos,i]=1
    end
    return result
end


function fprediction(circuit::QCircuit,data_set::Matrix{Float64})
    preds=fprediction_util(circuit,data_set)
    L=size(preds,2)
    result=zeros(Int,4,L)
    for i in 1:L
        pos=argmax(preds[:,i])
        result[pos,i]=1
    end
    return result
end


function train_circuits(nqubit,nlayer,mq,C,batchsize,train_datas,train_labels)
    circuits = Vector{Vector{Float64}}()
    alphas = Float64[]
    circuit = build_circuit(nqubit,nlayer,mq,1)
    params = parameters(circuit)
    statebatchs,labelbatchs=generate_batch(batchsize, train_datas, train_labels, isweight=true)
    for c in 1:C
        params=train_circuit(copy(params),statebatchs,labelbatchs)
        reset_parameters!(circuit,params)
        pred=fprediction(circuit,train_datas)
        result=fresult(train_labels,pred)
        W=get_weight(labelbatchs)
        ferror=result'*W
        alpha1=log((1-ferror)/ferror)+log(K-1)
        W.=W.*exp.(alpha1*result) 
        W./=sum(W)
        reset_weight!(labelbatchs,W)
        push!(alphas,alpha1)
        push!(circuits,params)
    end
    return circuits,alphas
end

function get_weight(labelbatchs)
    W=Float64[]
    for labels in labelbatchs
        W=vcat(W,vec(sum(labels,dims=1)))
    end
    return W
end
function reset_weight!(labelbatchs,W)
    pos=1
    for labels in labelbatchs
        for j in 1:size(labels,2)
            for k in 1:4
                if labels[k,j]!=0
                    labels[k,j]=W[pos]
                    pos+=1
                end
            end
        end
    end
end

function train_circuit(params,statebatchs,labelbatchs)
    opt=ADAM(lr)
    circuit=build_circuit(nqubit,nlayer,mq,1)
    reset_parameters!(circuit,params)
    for iter = 1:total_iter
        for (statebatch,labelbatch) in zip(statebatchs,labelbatchs)
            grad = gradient(x->loss_util(x, B, statebatch, labelbatch), circuit)[1]
            update!(opt, params,grad)
            reset_parameters!(circuit, params)
        end
    end
    return params
end


function collect_re(circuits,alphas,C,train_datas,test_datas,train_labels,test_labels)
    train_acc=zeros(C)
    test_acc=zeros(C)
    train_acc_tmp=zeros(4,1541)
    test_acc_tmp=zeros(4,726)
    circuit=build_circuit(nqubit,nlayer,mq,1)
    for j in 1:C
        reset_parameters!(circuit,circuits[j])
        train_acc_tmp+=fprediction_util(circuit,train_datas)*alphas[j]
        re1=fresult(train_labels,train_acc_tmp)
        train_acc[j]=1-sum(re1)/1541
        test_acc_tmp+=fprediction_util(circuit,test_datas)*alphas[j]
        re2=fresult(test_labels,test_acc_tmp)
        test_acc[j]=1-sum(re2)/726
    end
    return train_acc,test_acc
end

#分类类别
global const K=4
global nqubit=8
global nlayer=8
batchsize=1541
const total_iter=1500
lr=5e-3
mq=[7,8]
const B=[QubitsTerm(mq[1]=>UP),QubitsTerm(mq[2]=>UP)]
C=10
const ini_custate=CuStateVectorBatch(nqubit,batchsize)

train_datas = readdlm("../MNIST_data/Mnist_1357_train_set.csv",',')
train_labels = readdlm("../MNIST_data/Mnist_1357_train_labels.csv",',',Int)

test_datas = readdlm("../MNIST_data/Mnist_1357_test_set.csv",',')
test_labels = readdlm("../MNIST_data/Mnist_1357_test_labels.csv",',',Int)

statebatchs,labelbatchs=generate_batch(batchsize, train_datas, train_labels, isweight=true)

Sm=10

thetass = Vector{Vector{Vector{Float64}}}()
alphass = Vector{Vector{Float64}}()
for i in 1:Sm
    thetas,alphas=train_circuits(nqubit,nlayer,mq,C,batchsize,train_datas,train_labels)
    push!(thetass,thetas)
    push!(alphass,alphas)
end

test_acc=zeros(Sm,C)
train_acc=zeros(Sm,C)

for i in 1:Sm
    train_acc[i,:],test_acc[i,:]=collect_re(thetass[i],alphass[i],C,train_datas,test_datas,train_labels,test_labels)
end
println("pre_acc_train:$(mean(train_acc,dims=1))%;pre_acc_test:$(mean(test_acc,dims=1))%")
writedlm("ada_rotation_diff_classifier_acc_$(dev).csv",[train_acc,test_acc])
save("Mnist4_ada_rotation_thetas&alphas_$(dev).jld2","thetass",thetass,"alphass",alphass)



