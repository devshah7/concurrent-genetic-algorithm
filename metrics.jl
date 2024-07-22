module MetricsModule
import Statistics, YAML, Plots, Dates, StatsBase, Distances
using OrderedCollections
export update_evolution_metrics, plot_fitness, Metrics, save_index_file, calculate_basic_stats, plot_chrom_dists

function calculate_basic_stats(population)
    # Distances
    inds_1 = StatsBase.sample(population, length(population), replace=true)
    inds_2 = StatsBase.sample(population, length(population), replace=true)
    dists = [Distances.euclidean(inds_1[i].chromosome, inds_2[i].chromosome) for i = 1 : length(inds_1)]

    # Fitness
    fitnesses = [ind.fitness for ind in population]
    fitness_aggr = OrderedDict("mean_fitness" => Statistics.mean(fitnesses),
                               "max_fitness" => maximum(fitnesses),
                               "min_fitness" => minimum(fitnesses),
                               "std_fitness" => Statistics.std(fitnesses),
                               "mean_chromosome_dist" => Statistics.mean(dists),
                               "std_chromosome_dist" => Statistics.std(dists))
    return fitness_aggr
end

function update_evolution_metrics(metrics, evolution, num_total_evolutions, population)
    dir = joinpath(metrics.metrics_dir, "evolutions", "evolution_$(evolution)")
    mkpath(dir)

    fitness_aggr = calculate_basic_stats(population)
    YAML.write_file(joinpath(dir, "aggr_fitness.yml"), fitness_aggr)

    # Save info about individuals
    ind_info = Dict(ind.id => Dict("fitness" => ind.fitness, "chromosome" => ind.chromosome) for ind in population)
    YAML.write_file(joinpath(dir, "individuals.yml"), ind_info)

    # Print info
    println("===== Evolution $evolution / $num_total_evolutions Aggr Fitness Results =====")
    println(fitness_aggr)
    if fitness_aggr["mean_fitness"] > metrics.curr_best_avg_fitness
        println("NEW best avg fitness!")
        metrics.curr_best_avg_fitness = fitness_aggr["mean_fitness"]
        metrics.curr_best_avg_fitness_evolution = evolution
    end
    if fitness_aggr["max_fitness"] > metrics.curr_best_max_fitness
        println("NEW best max fitness!")
        metrics.curr_best_max_fitness = fitness_aggr["max_fitness"]
        metrics.curr_best_max_fitness_evolution = evolution
    end
    st = "Last best fitness => avg: $(metrics.curr_best_avg_fitness) (at $(metrics.curr_best_avg_fitness_evolution)) "
    st *= "&& max: $(metrics.curr_best_max_fitness) (at $(metrics.curr_best_max_fitness_evolution))"
    println(st)
    println("")

    metrics.last_avg_fitness = fitness_aggr["mean_fitness"]
    metrics.last_max_fitness = fitness_aggr["max_fitness"]
    metrics.last_fitness_evolution = evolution
    push!(metrics.evolutions, evolution)
    push!(metrics.fitness_means, fitness_aggr["mean_fitness"])
    push!(metrics.fitness_maxs, fitness_aggr["max_fitness"])
    push!(metrics.fitness_stds, fitness_aggr["std_fitness"])
    push!(metrics.chrom_dist_means, fitness_aggr["mean_chromosome_dist"])
    push!(metrics.chrom_dist_stds, fitness_aggr["std_chromosome_dist"])
end

function plot_fitness(metrics)
    dir = joinpath(metrics.metrics_dir, "plots")

    # Smooth max
    movingaverage(g, n) = [i < n ? Statistics.mean(g[begin:i]) : Statistics.mean(g[i-n+1:i]) for i in 1:length(g)]
    maxs = movingaverage(metrics.fitness_maxs, 5)

    y = hcat(metrics.fitness_means, maxs)
    y = Real.(y)
    ribbon = hcat(metrics.fitness_stds, zeros(Int, length(metrics.fitness_stds)))

    Plots.plot(metrics.evolutions, y, 
              grid=true, ribbon=ribbon, fillalpha=0.25, 
              title="Fitness by Evolution", legend=true,
              xlabel="evolution", ylabel="fitness (shaded std)",
              label = ["Avg Fitness" "Max Fitness"], lw=3)
    Plots.savefig(joinpath(dir, "fitness.png"))
end

function plot_chrom_dists(metrics)
    dir = joinpath(metrics.metrics_dir, "plots")

    y = metrics.chrom_dist_means
    ribbon = metrics.chrom_dist_stds

    Plots.plot(metrics.evolutions, y, 
              grid=true, ribbon=ribbon, fillalpha=0.25, 
              title="Avg Euclidean Chromosome Distance by Evolution", legend=false,
              xlabel="evolution", ylabel="distance (shaded std)", lw=3)
    Plots.savefig(joinpath(dir, "avg_euclidean_chromosome_distance.png"))
end

function save_index_file(metrics)
    dir = metrics.metrics_dir

    time_taken = Dates.canonicalize(Dates.CompoundPeriod(Dates.now() - metrics.starting_time))

    YAML.write_file(joinpath(dir, "index.yml"),
                    OrderedDict("best_avg_evolution" => metrics.curr_best_avg_fitness_evolution,
                                "best_avg_fitness" => metrics.curr_best_avg_fitness,
                                "best_max_evolution" => metrics.curr_best_max_fitness_evolution,
                                "best_max_fitness" => metrics.curr_best_max_fitness,
                                "last_evolution" => metrics.last_fitness_evolution,
                                "last_avg_fitness" => metrics.last_avg_fitness,
                                "last_max_fitness" => metrics.last_max_fitness,
                                "time_taken" => time_taken))
end

Base.@kwdef mutable struct Metrics
    evolutions = []
    fitness_means = []
    fitness_maxs = []
    fitness_stds = []
    chrom_dist_means = []
    chrom_dist_stds = []
    curr_best_avg_fitness = -Inf
    curr_best_avg_fitness_evolution = nothing
    curr_best_max_fitness = -Inf
    curr_best_max_fitness_evolution = nothing
    last_avg_fitness = nothing
    last_max_fitness = nothing
    last_fitness_evolution = nothing
    metrics_dir
    starting_time
end

end