include("./environment/objects/individual.jl")
include("./environment/game_engine.jl")
include("./environment/environment.jl")
include("./environment/neural_network.jl")
include("./genetic_algorithm.jl")
include("./metrics.jl")
include("./utils.jl")
include("./render_environment.jl")
using OrderedCollections, ArgParse
import YAML, Statistics, Dates, Distributions, Random
import .IndividualModule, .GameEngine, .NeuralNetwork, .Utils, .MetricsModule, .RenderEnvironment

function get_actual_evolution(train_experiment_name, evolution)
    if evolution in ["last", "best_max", "best_avg"]
        index = YAML.load_file(joinpath("outputs", train_experiment_name, "index.yml"))
        evolution = index["$(evolution)_evolution"]
    else
        evolution = parse(Int64, evolution)
    end
    return evolution
end

function save_args(experiment_path, train_experiment_name, evolution, top_n, jitter)
    args_dict = Dict("train_experiment_name" => train_experiment_name, "evolution" => evolution,
                     "top_n" => top_n, "jitter" => jitter)
    YAML.write_file(joinpath(experiment_path, "configs", "arguments.yml"), args_dict)
end

function fix_num_chromosomes(a, n)
    # Expand
    if length(a) < n
        num_needed_per = n // length(a)
        rem = n % length(a)
        new_a = []
        for i = 1 : length(a)
            num_needed = num_needed_per + rem * (i == 1)
            for _ = 1 : num_needed
                push!(new_a, a[i])
            end
        end
        return new_a
    # Shrink
    elseif length(a) > n
        return a[1 : n]
    end
    return a
end

function jitter_genes(a, frac)
    if frac === nothing
        return a
    end

    frac = parse(Float32, frac)
    new_a = []
    for chrom in a
        new_chrom = []
        for gene in chrom
            rand_val = Random.rand(Distributions.Uniform(-frac, frac))
            amt_to_change = rand_val * gene
            new_gene = gene + amt_to_change
            push!(new_chrom, convert(Float64, new_gene))
        end
        push!(new_a, Float64.(new_chrom))
    end
    return new_a
end

function load_individuals(config, train_experiment_name, evolution, nn_size, chromosome_length, next_obj_id, top_n, jitter)
    ind_info = YAML.load_file(joinpath("outputs", train_experiment_name, "evolutions", "evolution_$(evolution)", "individuals.yml"))
    # Collect/sort chromosomes
    chromosomes = [y[2]["chromosome"] for y in sort(collect(ind_info), by=x->x[2]["fitness"], rev=true)]
    # Select top n
    if top_n !== nothing
        top_n = parse(Int, top_n)
        chromosomes = chromosomes[1 : min(top_n, length(chromosomes))]
    end
    # Shrink/Expand out chromosomes if needed
    chromosomes = fix_num_chromosomes(chromosomes, config["population"]["pop_size"])
    # Jitter genes
    chromosomes = jitter_genes(chromosomes, jitter)
    @assert length(chromosomes) == config["population"]["pop_size"]
    population = IndividualModule.create_initial_population(config["population"], chromosome_length, nn_size, 
                                                            config["individual"], config["neural_network"],
                                                            next_obj_id,
                                                            chromosomes)
    return population
end

function save_metrics(experiment_path, population, train_experiment_name, evolution)
    fitness_aggr = MetricsModule.calculate_basic_stats(population)
    YAML.write_file(joinpath(experiment_path, "aggr_fitness.yml"), fitness_aggr)

    println("Aggr Fitness Results for Evaluation of Train Experiment: $(train_experiment_name) at evolution: $(evolution)")
    println(fitness_aggr)
    println("")
end

function evaluate(config, experiment_path, train_experiment_name, evolution, top_n, jitter, render)
    evolution = get_actual_evolution(train_experiment_name, evolution)
    save_args(experiment_path, train_experiment_name, evolution, top_n, jitter)

    next_obj_id = Utils.NextObjId(1)
    nn_size, chromosome_length = NeuralNetwork.calc_model_dims(config["individual"], config["neural_network"])
    population = load_individuals(config, train_experiment_name, evolution, nn_size, chromosome_length, next_obj_id, top_n, jitter)

    frames = GameEngine.run_episode(config["episode"], config["environment"], population,
                                    config["individual"]["missing_input_position_padding"],
                                    render)
    save_metrics(experiment_path, population, train_experiment_name, evolution)
    if render
        RenderEnvironment.render_environment(config["environment"], frames, experiment_path)
    end
end

function main()
    arg_parser = ArgParseSettings()
    @add_arg_table arg_parser begin
        "--experiment-name", "-n"
            help = "name of output experiment"
            required = false
            default = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
        "--train-experiment-name", "-t"
            help = "name of output experiment"
            required = false
            default = ""
        "--evolution", "-e"
            help = "evolution of train experiment to evaluate, such as last, best_avg, best_max or a number"
            required = false
            default = "last"
        "--override-config", "-o"
            help = "name of config to override the train experiment config with, optionally"
            required = false
            default = nothing
        "--top-n-chromosomes", "-c"
            help = "can specifiy this if wanting to run on only the top n chromosomes (expanded to be the size of the pop)"
            required = false
            default = nothing
        "--jitter-chromosome-fraction", "-j"
            help = "jitter the genes of each individuals chromosomes by some fraction of its original value"
            required = false
            default = nothing
        "--no-render", "-r"
            help = "render episode"
            action = :store_true
    end
    args = parse_args(arg_parser)

    if haskey(args, "train-experiment-name") == false
        println("WARNING: need a train experiment name to run evaluate.jl, closing...")
        return
    end

    experiment_name = args["experiment-name"]
    train_experiment_name = args["train-experiment-name"]
    evolution = args["evolution"]

    # Create experiment directory
    experiment_name = "EVALUATE-" * experiment_name
    experiment_path = joinpath("outputs", experiment_name)
    
    config_dir = joinpath(experiment_path, "configs")
    mkpath(config_dir)

    config_path = joinpath("outputs", train_experiment_name, "config.yml")
    config = OrderedDict(YAML.load_file(config_path))
    cp(config_path, joinpath(config_dir, "train_config.yml"), force=true)

    # Override config
    override_config_path = args["override-config"]
    if override_config_path !== nothing
        override_config_path = joinpath("configs", "$override_config_path.yml")
        cp(override_config_path, joinpath(config_dir, "override_config.yml"), force=true)
        Utils.override_config(config, YAML.load_file(override_config_path))
        YAML.write_file(joinpath(config_dir, "main_config.yml"), config)
    end

    render = args["no-render"] == false

    evaluate(config, experiment_path, train_experiment_name, evolution, args["top-n-chromosomes"],
             args["jitter-chromosome-fraction"], render)
end

main()