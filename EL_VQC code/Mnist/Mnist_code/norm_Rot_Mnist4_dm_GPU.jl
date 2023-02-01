using CUDA
push!(LOAD_PATH,"../../../package/QuantumCircuits_demo/src","../../../package/VQC_demo_cuda/src")
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
using Dates
dev=2
device!(dev)

function build_circuit(N,nlayer,batchsize,P)
    circuit = QCircuit()
    for k in 1:nlayer
        if k<9
            for j=1:N
                push!(circuit, ERyGate(j, rand(Float32,batchsize), batchsize))
            end
        end
        for j=1:N
            push!(circuit, RxGate(j, rand(Float32), isparas=true))
            push!(circuit, RzGate(j, rand(Float32), isparas=true))
            push!(circuit, RxGate(j, rand(Float32), isparas=true))
        end 
        for j=1:N-1
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
    set_data!(circuit,data_batch)
    tmp=circuit*ini_cudm
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

function generate_batch(batch_size, dataset, labels)
    L,N=size(dataset)
    data_batchs=Vector{Matrix}()
    label_batchs=Vector{Matrix}()
    pos=1
    while  pos <= N-batch_size+1
        push!(data_batchs,dataset[:,pos:pos+batch_size-1]) 
        push!(label_batchs,labels[:,pos:pos+batch_size-1])
        pos+=batch_size
    end
    # push!(data_batchs,dataset[:,pos:end])
    # push!(label_batchs,labels[:,pos:end])
    return data_batchs,label_batchs
end

function fresult(labels::Vector{Vector{Float64}},prediction::Matrix{Int})#判断分类的对错，分类正确：+；分类错误：-
    L=size(labels,1)
    result=zeros(L)
    for i in 1:L
        if labels[i]!=prediction[:,i]
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

function train_circuit(nqubit,nlayer,statebatchs,labelbatchs,P)
    opt=ADAM(lr)
    circuit=build_circuit(nqubit,nlayer,batchsize,P)
    params = parameters(circuit)
    for iter = 1:total_iter
        for (statebatch,labelbatch) in zip(statebatchs,labelbatchs)
            grad = gradient(x->loss_util(x, B, statebatch, labelbatch), circuit)[1]
            update!(opt, params,grad)
            reset_parameters!(circuit, params)
        end
    end
    return params
end


#分类类别
global const K=4
global nqubit=8
global nlayer=20

total_iter=1000
lr=5e-3

train_datas = readdlm("../MNIST_data/Mnist_1357_train_set.csv",',',Float32)
train_labels = readdlm("../MNIST_data/Mnist_1357_train_labels.csv",',',Int)

test_datas = readdlm("../MNIST_data/Mnist_1357_test_set.csv",',',Float32)
test_labels = readdlm("../MNIST_data/Mnist_1357_test_labels.csv",',',Int)

batchsize=256
const ini_cudm=CuDensityMatrixBatch{ComplexF32}(nqubit,batchsize)
statebatchs,labelbatchs=generate_batch(batchsize, train_datas, train_labels)

#测量算子
global mq=[7,8] #measure_qubits
const B=[QubitsTerm(mq[1]=>UP),QubitsTerm(mq[2]=>UP)]

Ps=convert.(Float32,[0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12,0.14,0.16,0.18])
Sm=3

# f=jldopen("norm_rot_data/norm_rot_dm_20_$(dev).jld2","a+")
# f["config/batch"]=batchsize
# f["config/iter"] = total_iter
# f["config/nqubit"]=nqubit
# f["config/nlayer"]=nlayer
# f["config/lr"]=lr
# f["config/mq"]=mq
# close(f)
# for j in dev*3+1:(dev+1)*3
#     P=Ps[j]
#     circuit=build_circuit(nqubit,nlayer,1,P)
#     for i in 1:Sm
#         theta= train_circuit(nqubit,nlayer,statebatchs,labelbatchs,P)
#         reset_parameters!(circuit,theta)
#         x1=fprediction(circuit,train_datas)
#         train_acc=dot(x1,train_labels)/1541
#         x2=fprediction(circuit,test_datas)
#         test_acc=dot(x2,test_labels)/726
#         println("Train_acc:$(train_acc);Test_acc:$(test_acc)")
#         f=jldopen("norm_rot_data/norm_rot_dm_20_$(dev).jld2","a+")
#         f["$(P)/thetas/theta$(i)"]=theta
#         f["$(P)/acc/train$(i)"] = train_acc
#         f["$(P)/acc/test$(i)"] = test_acc
#         close(f)
#         open("norm_rot_data/$(dev)_process.txt","a+") do io
#         writedlm(io,["dev:$(dev),$(P):训练完成第$(i)个,剩余$((4-j)*Sm-i).完成时间：$(now())"])
#         end
# end
# end
#writedlm("norm_rotation_difflayer_theta.csv",theta)
#writedlm("norm_rotation_difflayer_acc_4.csv",[train_acc,test_acc])