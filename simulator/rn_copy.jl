using Catalyst
using DelaySSAToolkit
using Distributions
using Random

# Marcov Part
# G_on -> G_on + N
# N -> M
# N -> 0
# M -> 0
# G_mid -> G_off

# set parameters
n = 100 # number of samples
ts = [1, 2, 5, 10, 15] # time points
rate_ranges = [ 0 50
                0 30
                0 3
                0 4 
                0 10
                0 1
                0 3
                0 3] # ranges for r1, r2, r3, r4, a1, b1, a2, b2
u0 = [0.0, 0.0, 0.0, 1.0, 0.0] # initial condition for N, M, G_on, G_mid, G_off
rng = MersenneTwister(2024) # random number generator

# Marcov Part
rn = @reaction_network begin
    @parameters r1 r2 r3 r4 
    @species N(t) M(t) G_on(t) G_mid(t) G_off(t)
    r1*G_on, 0 --> N
    r2, N --> M
    r3, N --> 0
    r4, M --> 0
    1000*G_mid, G_mid --> G_off
end

# Non-Marcov Part
de_chan0 = [[],[]]
delay_trigger_affect_raw! = function (integrator, gamma_p, rng)
    """ Non-Marcov G_on->G_off & G_off->G_on delay_affect
    gamma_p: parameters of Gamma distributions
    rng: random number generator
 """
    d1 = Gamma(gamma_p[1], gamma_p[2])
    d2 = Gamma(gamma_p[3], gamma_p[4])
    τ1 = rand(rng, d1)
    τ2 = rand(rng, d2)
    append!(integrator.de_chan[1], τ2)
    append!(integrator.de_chan[2], τ1+τ2)
end

delay_complete = Dict(1 => [3 => 1, 5 => -1], 2 => [3 => -1, 4 => 1])
# de_channel[1] 触发, G_off->G_on; de_channel[2] 触发, G_on->G_mid

delay_interrupt = Dict()
# 没有中断机制

println("reaction network established.")

