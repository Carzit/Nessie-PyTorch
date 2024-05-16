# using Logging
# disable_logging(Logging.Warn)

using Pkg
Pkg.activate("env")

using Sobol, JSON
using ProgressMeter

include("rn_copy.jl")
# What parameters can be set in rn.jl?
# n: number of samples
# ts: time points
# rate_ranges: ranges for r1, r2, r3, r4
# gamma_p: parameters of Gamma distributions
# u0: initial condition for N, M, G_on, G_mid, G_off
# rng: random number generator

# set parameters
n_traj = Int(1000)  # number of trajectories in a single ssa simulation
marginals = [1, 2]  # 1 for N, 2 for M
saveat = 0.1        # saveat
max_copy_num = Int(100)  # 超过max_copy_num(100)的copynum在计算概率密度时会被视作100

# initial computing object
jumpsys = convert(JumpSystem, rn; combinatoric_ratelaws=false)
aggregatoralgo = DelayDirect()
tspan = (0.0, last(ts))

# define solver
function solver(p, n_traj, ts; marginals=[1, 2])
    # 求解
    delay_trigger_affect!(integrator, rng) = delay_trigger_affect_raw!(integrator, p[5:8], rng)
    delay_trigger = Dict(5 => delay_trigger_affect!)
    delaysets = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt)
    dprob = DiscreteProblem(u0, tspan, p[1:4])
    djprob = DelayJumpProblem(
        jumpsys,
        dprob,
        aggregatoralgo,
        delaysets,
        de_chan0;
        save_positions=(false, false),
        save_delay_channel=false,
    )
    ens_prob = EnsembleProblem(djprob)
    ens = solve(
    ens_prob, SSAStepper(); trajectories=n_traj, saveat=saveat
    )
    # 提取概率密度
    t_index = [Int(round(t/saveat+1)) for t in ts]
    sol_int = map(x -> Int(round(x)), ens[marginals, :, :])
    pd = Vector{Vector{Vector{Float64}}}() #第一层时间，第二层[[copy_num, prob_1, prob_2]]
    for j in eachindex(t_index)
        push!(pd, Float64[])        
        for i in 0:max_copy_num-1
            temp = Vector{Float64}([Float64(i)])
            for marg in marginals
                push!(temp, Float64(count(x -> x == i, sol_int[marg, t_index[j], :])/n_traj))
            end
            push!(pd[j], temp)
        end
        temp = Vector{Float64}([Float64(max_copy_num)])
        for marg in marginals
            push!(temp, Float64(count(x -> x >= max_copy_num, sol_int[marg, t_index[j], :]/n_traj)))
        end
        push!(pd[j], temp)
    end
    return pd
end

# define parallel_simulate
function parallel_simulate(ts, ps, n_traj; marginals=[1, 2])
    progress = Progress(length(ps), 1, "Generating data... ")
    X = Vector{Vector{Vector{Float64}}}() # 第一层线程数，第二层采样点数，第三层为数据向量
    y = Vector{Vector{Vector{Vector{Float64}}}}() # 第一层线程数，第二层采样点数，第三层为[[num of copy, prob N, prob M]]
    for i in 1:Threads.nthreads()
        push!(X, Float64[])
        push!(y, Float64[])
    end
    Threads.@threads for p in ps
        ProgressMeter.next!(progress)
        ret = solver(p, n_traj, ts; marginals=marginals)
        for i in eachindex(ret)
            push!(X[Threads.threadid()], [ts[i], p...])
            push!(y[Threads.threadid()], ret[i])
        end
    end
    vcat(X...), vcat(y...)
end

# generate ranging parameters
seq = SobolSeq(rate_ranges[:,1], rate_ranges[:,2])
ps = [ Sobol.next!(seq) for i in 1:n]

# generate data
data_X, data_y = parallel_simulate(ts, ps, n_traj; marginals=marginals)

# save data
json_data = Dict("data_X" => data_X, "data_Y" => data_y)
json_str = JSON.json(json_data)
open("data_dssa.json","w") do io
    write(io, json_str)
end
