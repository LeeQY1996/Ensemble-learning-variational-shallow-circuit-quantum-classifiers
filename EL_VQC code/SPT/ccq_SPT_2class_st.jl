push!(LOAD_PATH,"../../../package/QuantumCircuits_demo/src","../../../package/VQC_demo_cuda/src")
using VQC
using QuantumCircuits, QuantumCircuits.Gates
using DelimitedFiles
using Random
using Flux.Optimise:ADAM,update!
using Base.Threads
using Statistics
using StatsBase
using Zygote
using Plots


#label: one hot encoding

function load_Qdata(datas,labels)
    L,N=size(datas)
    #labels one-hot vector
    labels_vector=zeros(N,K)
    for i in 1:N
        pos=Int(labels[i])
        labels_vector[i,pos+1]=1
    end
    data_set=[[StateVector(datas[:,i]),labels_vector[i,:]] for i in 1:N]
    return data_set
end

function build_circuit(N,nlayer)
    ctrl_seed=rand(1:N-1,nlayer)
    circuit = QCircuit()
    for k in 1:nlayer
        for j=1:N
            push!(circuit, RxGate(j, rand(), isparas=true))
            push!(circuit, RzGate(j, rand(), isparas=true))
            push!(circuit, RxGate(j, rand(), isparas=true))
        end 
        for j=1:N
            if (j+ctrl_seed[k])%N ==0
                index=N
            else
                index=(j+ctrl_seed[k])%N
            end
            push!(circuit, CRxGate(index,j, rand(), isparas=true))   
            push!(circuit, CRzGate(index,j, rand(), isparas=true))  
            push!(circuit, CRxGate(index,j, rand(), isparas=true))    
        end
    end
    push!(circuit, RxGate(5, rand(), isparas=true))
    push!(circuit, RzGate(5, rand(), isparas=true))
    push!(circuit, RxGate(5, rand(), isparas=true))
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

function loss_util(circuit,data_batch)
    loss_value = 0.
    L=length(data_batch)
    for i in 1:L
        x,y=data_batch[i]
        p01=real(expectation(B[1],circuit*x))[1]
        p1=p01
        p2=(1-p01)
        loss_value-=(y[1]*log(p1)+y[2]*log(p2))
    end
    return loss_value/L
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

function fprediction_util(circuit::QCircuit,data_set)#输出预测向量.
    L=length(data_set)
    preds=zeros(L,2)
    for i in 1:L
        x=data_set[i][1]
        p01=real(expectation(B[1],circuit*x))[1] #测量第一个比特0的概率
        preds[i,:]=[p01,1-p01]
    end
    return preds
end

function fprediction_util(circuits::Vector{QCircuit},alphas::Vector{Float64},data_set)
    L=length(data_set)
    preds=zeros(L,2)
    for i in 1:length(alphas)
        preds.+=fprediction_util(circuits[i],data_set).*alphas[i]
    end
    return preds
end

function fprediction(circuits::Vector{QCircuit},alphas::Vector{Float64},data_set)
    preds=fprediction_util(circuits,alphas,data_set)
    L=size(preds,1)
    result=zeros(Int,L,2)
    for i in 1:L
        pos=argmax(preds[i,:])
        result[i,pos]=1
    end
    return result
end

function fprediction_util(circuits::Vector{QCircuit},alphas::Vector{Float64},data_set::Matrix{Float64})
    L=size(data_set,2)
    preds=zeros(L,K)
    for i in 1:length(alphas)
        preds.+=fprediction_util(circuits[i],data_set).*alphas[i]
    end
    return preds
end

function fprediction_util(circuit::QCircuit,data_set::Matrix{Float64})
    L=size(data_set,2)
    preds=zeros(L,K)
    for i in 1:L
        x=circuit*StateVector(data_set[:,i])
        p=real(expectation(B[1],x))[1] #测量第一个比特0的概率
        preds[i,:]=[p,(1-p)]
    end
    return preds
end

#分类类别
global const K=2

global N=9
total_iter=1000
lr=0.005
global nlayer=3

#测量算子
UP=[1 0; 0 0]
DOWN=[0 0; 0 1]
global mq=[5] #measure_qubits
const B=[QubitsTerm(mq[1]=>UP),QubitsTerm(mq[1]=>"X"),QubitsTerm(mq[1]=>DOWN)]

datas = readdlm("SPT_train_set_$(N).csv",',', Float64)
labels = readdlm("SPT_cnn_train_label.csv",',', Float64)
data_set = load_Qdata(datas,labels)

test_set = readdlm("SPT_test_set_$(N).csv",',', Float64)

labels_train=[y for (x,y) in data_set]
batch=40

ress=zeros(4096)
for i in 1:10
    opt=ADAM(lr)
    circuit=build_circuit(N,nlayer)
    params = parameters(circuit)
    for iter = 1:total_iter
        data_batchs=generate_batch(data_set,batch)
        for data_batch in data_batchs
        grad = gradient(x->loss_util(x,data_batch),circuit)[1]
        update!(opt, params,grad)
        reset_parameters!(circuit, params)
        end
    end

    # x1=fprediction([circuit],[1.],data_set)
    # re1=fresult(labels_train,x1)
    # println("pre_acc_train:$(1-sum(re1)/40)%")

    # re=zeros(64,64)
    # for i in 1:64
    #     for j in 1:64
    #             tmp=circuit*StateVector(test_set[:,i*64+j-64])
    #             re[j,i]-=expectation(B[1],tmp)
    #     end
    # end
    res=fprediction([circuit],[1.],test_set)
    for i in 1:4096
        if res[i,1]==0
            ress[i] += 1
        else
            ress[i] += 0
        end
    end
end
ress=reshape(ress,64,64)./10


h1_vals = [0.1000, 0.2556, 0.4111, 0.5667, 0.7222, 0.8778, 1.0333, 1.1889, 1.3444, 1.5000]
anti_ferro_mag_boundary = [-1.004, -1.0009, -1.024, -1.049, -1.079, -1.109, -1.154,  -1.225, -1.285, -1.35]
para_mag_boundary = [0.8439, 0.6636, 0.5033, 0.3631, 0.2229, 0.09766, -0.02755, -0.1377, -0.2479, -0.3531]

# heatmap(LinRange(0,1.6,64),LinRange(-1.6,1.6,64),re,c=cgrad([:green,:yellow]))
# plot!(h1_vals,anti_ferro_mag_boundary,markershapes=:diamond,markersize=5,lw=3,linestyle=:dash,lc="red",mc=:purple,fc="white")
# plot!(h1_vals,para_mag_boundary,markershapes=:diamond,markersize=5,lw=3,linestyle=:dash,lc="blue",mc=:orange,fc="white")

heatmap(LinRange(0,1.6,64),LinRange(-1.6,1.6,64),ress,c=cgrad([:green,:yellow]))
plot!(h1_vals,anti_ferro_mag_boundary,markershapes=:diamond,markersize=5,lw=3,linestyle=:dash,lc="red",mc=:purple,fc="white")
plot!(h1_vals,para_mag_boundary,markershapes=:diamond,markersize=5,lw=3,linestyle=:dash,lc="blue",mc=:orange,fc="white")
annotate!(0.75,1,"Paramagnetic",color="blue")
annotate!(0.75,-1.5,"Antiferromagnetic",color="red")
annotate!(0.75,-0.5,"SPT",color="red")