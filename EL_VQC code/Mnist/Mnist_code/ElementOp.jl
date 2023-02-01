using Random

function build_circuit(nqubit::Int,nlayer::Int,mq::Vector{Int})
    circuit = QCircuit()
    for k in 1:nlayer
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

function build_circuit(nqubit::Int,nlayer::Int,mq::Vector{Int},P::Float64)
    circuit = QCircuit()
    for k in 1:nlayer
        for j=1:nqubit
            push!(circuit, RxGate(j, rand(), isparas=true))
            push!(circuit, RzGate(j, rand(), isparas=true))
            push!(circuit, RxGate(j, rand(), isparas=true))
        end 
        for j in 1:nqubit-1
            push!(circuit, CNOTGate(j,j+1))
            push!(circuit, Depolarizing(j, p=P))
            push!(circuit, Depolarizing(j+1, p=P))
        end
    end
    for j in mq
        push!(circuit, RxGate(j, rand(), isparas=true))
        push!(circuit, RzGate(j, rand(), isparas=true))
        push!(circuit, RxGate(j, rand(), isparas=true))
    end
    return circuit
end

# function load_QAdata(datas,labels,weight::Bool=false,Dm::Bool=false)
#     N,L=size(datas)
#     data_set=[]
#     for i in 1:L
#         Dm ? state = DensityMatrix(StateVector(datas[:,i])) : state = StateVector(datas[:,i])
#         weight ? el = [state,labels[:,i],1/L] : el = [state,labels[:,i]]
#         push!(data_set,el)
#     end
#     return data_set
# end

# function load_Qdata(datas,labels,weight::Bool=false)
#     N,L=size(datas)
#     data_set=[]
#     for i in 1:L
#         state = datas[:,i]
#         weight ? el = [state,labels[:,i],1/L] : el = [state,labels[:,i]]
#         push!(data_set,el)
#     end
#     return data_set
# end



# function load_QRdata(datas,labels,weight::Bool=false,Dm::Bool=false)
#     N,L=size(datas)
#     circuit=encode_circuit(N)
#     state=StateVector(N)
#     data_set=[]
#     for i in 1:L
#         Dm ? state = DensityMatrix(encode(circuit,datas[:,i],state)) : state = encode(circuit,datas[:,i],state)
#         weight ? el = [state,labels[:,i],1/L] : el = [state,labels[:,i]]
#         push!(data_set,el)
#     end
#     return data_set
# end
# function encode_circuit(Nqubits)
#     circuit=QCircuit()
#     for i in 1:Nqubits
#         push!(circuit,RyGate(i,rand(),isparas=true))
#     end
#     return circuit
# end

# function encode(circuit,params,state)
#     reset_parameters!(circuit,params)
#     return circuit*state
# end


# function generate_batch(train_set,batch_size)
#     L=length(train_set)
#     tmp=train_set
#     pos=1
#     data_batchs=[]
#     while pos < L-batch_size
#         push!(data_batchs,tmp[pos:pos+batch_size-1])
#         pos+=batch_size
#     end 
#     push!(data_batchs,tmp[pos:end])
#     return data_batchs
# end

# function generate_batch(nqubit, batch_size, dataset, labels)
#     L,N=size(dataset)
#     data_batchs=Vector{StateVectorBatch}()
#     label_batchs=Vector{Matrix}()
#     pos=1
#     while pos < N-batch_size
#         push!(data_batchs,StateVectorBatch(dataset[:,pos:pos+batch_size-1],nqubit,batch_size))
#         push!(label_batchs,labels[:,pos:pos+batch_size-1])
#         pos+=batch_size
#     end
#     push!(data_batchs,StateVectorBatch(dataset[:,pos:end],nqubit,N-pos+1))
#     push!(label_batchs,labels[:,pos:end])
#     return data_batchs,label_batchs
# end

function generate_batch(nqubit, batch_size, dataset, labels; isdm::Bool=false, isweight::Bool=false)
    L,N=size(dataset)
    isdm ? data_batchs=Vector{DensityMatrixBatch}() : data_batchs=Vector{StateVectorBatch}()
    label_batchs=Vector{Matrix}()
    weight_batchs=Vector{Vector}()
    if isweight
        weight_batchs=Vector{Vector}()
        We=ones(N)./N
    end
    pos=1
    while pos < N-batch_size
        isdm ? push!(data_batchs,DensityMatrixBatch(todm(dataset[:,pos:pos+batch_size-1]),nqubit,batch_size)) : push!(data_batchs,StateVectorBatch(dataset[:,pos:pos+batch_size-1],nqubit,batch_size))
        push!(label_batchs,labels[:,pos:pos+batch_size-1])
        isweight && push!(weight_batchs,We[pos:pos+batch_size-1])
        pos+=batch_size
    end
    # isdm ? push!(data_batchs,DensityMatrixBatch(todm(dataset[:,pos:end]),nqubit,N-pos+1)) : push!(data_batchs,StateVectorBatch(dataset[:,pos:end],nqubit,N-pos+1))
    # push!(label_batchs,labels[:,pos:end])
    # isweight && push!(weight_batchs,We[pos:end])
    if isweight
         return data_batchs,label_batchs,weight_batchs 
    else
        return data_batchs,label_batchs
    end
end

function generate_batch(nqubit, batch_size, dataset, labels; isdm::Bool=false, isweight::Bool=false)
    L,N=size(dataset)
    isdm ? data_batchs=Vector{DensityMatrixBatch}() : data_batchs=Vector{StateVectorBatch}()
    label_batchs=Vector{Matrix}()
    weight_batchs=Vector{Vector}()
    if isweight
        weight_batchs=Vector{Vector}()
        We=ones(N)./N
    end
    pos=1
    while pos < N-batch_size
        isdm ? push!(data_batchs,DensityMatrixBatch(todm(dataset[:,pos:pos+batch_size-1]),nqubit,batch_size)) : push!(data_batchs,StateVectorBatch(dataset[:,pos:pos+batch_size-1],nqubit,batch_size))
        push!(label_batchs,labels[:,pos:pos+batch_size-1])
        isweight && push!(weight_batchs,We[pos:pos+batch_size-1])
        pos+=batch_size
    end
    # isdm ? push!(data_batchs,DensityMatrixBatch(todm(dataset[:,pos:end]),nqubit,N-pos+1)) : push!(data_batchs,StateVectorBatch(dataset[:,pos:end],nqubit,N-pos+1))
    # push!(label_batchs,labels[:,pos:end])
    # isweight && push!(weight_batchs,We[pos:end])
    if isweight
         return data_batchs,label_batchs,weight_batchs 
    else
        return data_batchs,label_batchs
    end
end

function cugenerate_batch(nqubit, batch_size, dataset, labels; isdm::Bool=false, isweight::Bool=false)
    L,N=size(dataset)
    isdm ? data_batchs=Vector{CuDensityMatrixBatch}() : data_batchs=Vector{CuStateVectorBatch}()
    label_batchs=Vector{Matrix}()
    weight_batchs=Vector{Vector}()
    if isweight
        weight_batchs=Vector{Vector}()
        We=ones(N)./N
    end
    pos=1
    while pos < N-batch_size
        isdm ? push!(data_batchs,CuDensityMatrixBatch(CuArray(todm(dataset[:,pos:pos+batch_size-1])),nqubit,batch_size)) : push!(data_batchs,CuStateVectorBatch(CuArray(dataset[:,pos:pos+batch_size-1]),nqubit,batch_size))
        push!(label_batchs,labels[:,pos:pos+batch_size-1])
        isweight && push!(weight_batchs,We[pos:pos+batch_size-1])
        pos+=batch_size
    end
    # isdm ? push!(data_batchs,CuDensityMatrixBatch(CuArray(todm(dataset[:,pos:end]),nqubit,N-pos+1))) : push!(data_batchs,CuStateVectorBatch(CuArray(dataset[:,pos:end]),nqubit,N-pos+1))
    # push!(label_batchs,labels[:,pos:end])
    # isweight && push!(weight_batchs,We[pos:end])
    if isweight
         return data_batchs,label_batchs,weight_batchs 
    else
        return data_batchs,label_batchs
    end
end

function todm(x::AbstractMatrix{T}) where T
    L,N=size(x)
    tmp=zeros(T,L^2*N)
    for i in 1:N
        tmp[(i-1)*L^2+1:i*L^2].=vec(x[:,i]*x[:,i]')
    end
    return tmp
end

function fresult(labels::Matrix{Int},prediction::Matrix{Int})
    L=size(labels,1)
    result=ones(L)
    for i in 1:L
        result[i]=dot(labels[:,i],prediction[:,i])
    end
    return result
end

# function replace_params(data,params)
#     params[1:8]=data[1:8]
#     params[33:40]=data[9:16]
#     return params
# end

# function fprediction_util(circuit::QCircuit,data_set,K)#输出预测向量.
#     L=length(data_set)
#     preds=zeros(K,L)
#     params=parameters(circuit)
#     for i in 1:L
#         x=data_set[i][1]
#         params=replace_params(x,params)
#         reset_parameters!(circuit,params)
#         p01=real(expectation(B[1],circuit*StateVector(8)))[1] #测量第一个比特0的概率
#         preds[:,i]=[p01,1-p01]
#     end
#     return preds
# end

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


function loss_util(circuit::QCircuit, B::Vector{QubitsTerm}, data_batch::CuStateVectorBatch, label_batch::Matrix{<:Number})
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
function loss_util(circuit::QCircuit, B::Vector{QubitsTerm}, data_batch::CuDensityMatrixBatch, label_batch::Matrix{<:Number})
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