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

function build_circuit(nqubit,nlayer,mq,batchsize,P)
    circuit = QCircuit()
    for k in 1:nlayer
        if k<9
            for j=1:nqubit
                push!(circuit, ERyGate(j, rand(Float32,batchsize), batchsize))
            end
        end
        for j=1:nqubit
            push!(circuit, RxGate(j, rand(Float32), isparas=true))
            push!(circuit, RzGate(j, rand(Float32), isparas=true))
            push!(circuit, RxGate(j, rand(Float32), isparas=true))
        end 
        for j=1:nqubit-1
            push!(circuit,CNOTGate(j,j+1))
            push!(circuit, Depolarizing(j, p=P))
            push!(circuit, Depolarizing(j+1, p=P))
        end
    end
    for j in mq
    push!(circuit, RxGate(j, rand(Float32), isparas=true))
    push!(circuit, RzGate(j, rand(Float32), isparas=true))
    push!(circuit, RxGate(j, rand(Float32), isparas=true))
    end
    return circuit
end

function set_data!(circuit::QCircuit,data::Matrix{Float32})
    pos=1
    b=size(data,2)
    for j in 1:length(circuit)
        if typeof(circuit[j]) <: ERyGate
            circuit[j].paras=data[pos,:]
            circuit[j].batch=(b,)
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
    state=CuDensityMatrixBatch{ComplexF32}(nqubit,L)
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


function train_circuits(nqubit,nlayer,mq,C,batchsize,train_datas,train_labels,P)
    circuits = Vector{Vector{Float64}}()
    alphas = Float64[]
    circuit = build_circuit(nqubit,nlayer,mq,1,P)
    params = parameters(circuit)
    statebatchs,labelbatchs=generate_batch(batchsize, train_datas, train_labels,isweight=true)
    for c in 1:C
        params=train_circuit(copy(params),statebatchs,labelbatchs,P)
        reset_parameters!(circuit,params)
        pred=fprediction(circuit,train_datas)
        result=fresult(train_labels,pred)[1:1536]
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

function train_circuit(params,statebatchs,labelbatchs,P)
    opt=ADAM(lr)
    circuit=build_circuit(nqubit,nlayer,mq,1,P)
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


function collect_re(circuits,alphas,C,train_datas,test_datas,train_labels,test_labels,P)
    train_acc=zeros(C)
    test_acc=zeros(C)
    train_acc_tmp=zeros(4,1541)
    test_acc_tmp=zeros(4,726)
    circuit=build_circuit(nqubit,nlayer,mq,1,P)
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
batchsize=256
const total_iter=500
lr=5e-3
mq=[7,8]
UP1=convert.(Float32,UP)
const B=[QubitsTerm(mq[1]=>UP1),QubitsTerm(mq[2]=>UP1)]
C=5
const ini_custate=CuDensityMatrixBatch{ComplexF32}(nqubit,batchsize)

train_datas = readdlm("../MNIST_data/Mnist_1357_train_set.csv",',',Float32)
train_labels = readdlm("../MNIST_data/Mnist_1357_train_labels.csv",',',Int)

test_datas = readdlm("../MNIST_data/Mnist_1357_test_set.csv",',',Float32)
test_labels = readdlm("../MNIST_data/Mnist_1357_test_labels.csv",',',Int)

statebatchs,labelbatchs=generate_batch(batchsize, train_datas, train_labels, isweight=true)

Sm = 2

Ps=convert.(Float32,[0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12,0.14,0.16,0.18])
for j in dev+1:dev+1
    P=Ps[j]
    for i in 1:Sm
        circuits,alphas=train_circuits(nqubit,nlayer,mq,C,batchsize,train_datas,train_labels,P)
        f=jldopen("Ada_Rot_data/$(P)_ada3_Rot_dm_cu_c&a.jld2","a+")
        f["circuits$(i)"] = circuits
        f["alphas$(i)"] = alphas
        close(f)
        train_acc,test_acc=collect_re(circuits,alphas,C,train_datas,test_datas,train_labels,test_labels,P)
        open("Ada_Rot_data/$(P)_ada3_amplitude_cu_acc_c$(C)s$(Sm).csv","a+") do io
            writedlm(io,[train_acc,test_acc])
        end
        open("Ada_Rot_data/$(P)_process.txt","a+") do io
            writedlm(io,["$(P):训练完成第$(i)个,剩余$(Sm-i)."])
        end
        println("$(P):训练完成第$(i)个,剩余$(Sm-i).")
    end
end
# re=load("Mnist4_ada_rotation_thetas&alphas_0.jld2")
# thetass=re["thetass"]
# alphass=re["alphass"]
# train_acc=zeros(10,10,12)
# test_acc=zeros(10,10,12)
# for j in dev*3+1:(dev+1)*3
#     P=Ps[j]
#     for i in 1:10
#         circuits,alphas=thetass[i],alphass[i]
#         train_acc[:,i,j],test_acc[:,i,j]=collect_re(circuits,alphas,10,train_datas,test_datas,train_labels,test_labels,P)
#         println("$(P):训练完成第$(i)个,剩余$(10-i).")
#     end
# end


