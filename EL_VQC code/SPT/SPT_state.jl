using Distributed
PID = addprocs(1)
@everywhere push!(LOAD_PATH,"../../package/Meteor/src")
@everywhere using Meteor
using DelimitedFiles



global N=15

@everywhere function SPTmodel(N::Real,J::Real,h1::Real,h2::Real)
    H=QubitsOperator()
    for i in 1:N-2
        H+=QubitsTerm(i=>"Z", i+1=>"X",i+2=>"Z", coeff=-J)
    end
    for i in 1:N
        H+=QubitsTerm(i=>"X",coeff=-h1)
    end
    for i in 1:N-1
        H+=QubitsTerm(i=>"X",i+1=>"X",coeff=-h2)
    end
    return H
end

@everywhere function solver_SPTmodel(N::Real,J::Real,h1::Real,h2::Real)
    H=SPTmodel(N,J,h1,h2)
    e,v=ground_state(H)
    return e,v
end

# J=1
# #生成训练集
# h1=LinRange(0,2,40)
# eigenvalues=zeros(40)
# eigenstates=zeros(2^N,40)
# for i in 1:40
#     H=SPTmodel(N,J,h1[i],0)
#     e,v=ground_state(H)
#     eigenvalues[i]=e
#     eigenstates[:,i]=v
# end
# writedlm("SPT_train_set_$(N).csv",eigenstates,',')
# labels=ones(40)
# labels[21:40].=0
# writedlm("SPT_train_label_$(N).csv",labels,',')


# for i in 1:64
#     for j in 1:64
#     push!(Fetch,@spawn solver_SPTmodel(N,J,h1[i],h2[j]))
#     end
# end

# for i in Fetch
#     circuits,alphas=fetch(i)
#     push!(circuitss,circuits)
#     push!(alphass,alphas)
# end

#生成测试集
J=1
h1=LinRange(0,1.6,64)
h2=LinRange(-1.6,1.6,64)
eigenvalues=zeros(64^2)
eigenstates=zeros(2^N,64^2)
# for i in 1:64
#     for j in 1:64
#         H=SPTmodel(N,J,h1[i],h2[j])
#         e,v=ground_state(H)
#         eigenvalues[i*j]=e
#         eigenstates[:,(i-1)*64+j]=v
#     end
# end
Fetch=[]
for i in 1:64
    for j in 1:64
    push!(Fetch,@spawn solver_SPTmodel(N,J,h1[i],h2[j]))
    end
end

for i in 1:64
    for j in 1:64
        eigenstates[:,(i-1)*64+j]=fetch((i-1)*64+j)[2]
    end
end

writedlm("SPT_test_set_$(N).csv",eigenstates,',')


