include("./environment/objects/individual.jl")
include("./environment/game_engine.jl")
include("./environment/environment.jl")
include("./environment/neural_network.jl")
include("./genetic_algorithm.jl")
include("./metrics.jl")
include("./utils.jl")
using ArgParse
import YAML, Flux, .Threads, Plots, Dates
import .IndividualModule, .GameEngine, .GeneticAlgorithm, .MetricsModule, .NeuralNetwork, .Utils

function create_population(population_config, ga_config, chromosome_length, 
                           nn_size, individual_config, neural_network_config, next_obj_id,
                           population=nothing)
    # Either creates initial population or evolves current population
    if population === nothing
        population = IndividualModule.create_initial_population(population_config, chromosome_length, nn_size, 
                                                                individual_config, neural_network_config, next_obj_id)
    else
        population = GeneticAlgorithm.evolve_population(ga_config, population, nn_size, individual_config, 
                                                        neural_network_config, next_obj_id)
    end
    # Reset certain things for each individual
    IndividualModule.reset_population_before_episode(population)
    return population
end

function train(config, experiment_path)
    # Create a obj of current id available for new objects
    next_obj_id = Utils.NextObjId(1)

    # Run evolutions
    evolutions_config = config["evolutions"]
    num_total_evolutions = evolutions_config["num_total_evolutions"]
    population = nothing
    metrics = MetricsModule.Metrics(metrics_dir=experiment_path, starting_time=Dates.now())
    nn_size, chromosome_length = NeuralNetwork.calc_model_dims(config["individual"], config["neural_network"])
    for evolution = 1 : num_total_evolutions
        # Evolve population OR create initial population if first evolution
        population = create_population(config["population"], config["ga"], chromosome_length, 
                                       nn_size,
                                       config["individual"], config["neural_network"],
                                       next_obj_id, population)
        # Run episode from game engine (evolution of environment)
        GameEngine.run_episode(config["episode"], config["environment"], population,
                                config["individual"]["missing_input_position_padding"])
        # Save metrics on population fitness and gene info
        MetricsModule.update_evolution_metrics(metrics, evolution,
                                                num_total_evolutions, population)
    end
    
    # Save index file for best and last evolution information
    MetricsModule.save_index_file(metrics)

    # Plot avg fitness over evolutions
    MetricsModule.plot_fitness(metrics)
    # And distances of chromosomes
    MetricsModule.plot_chrom_dists(metrics)
end

function main()
    arg_parser = ArgParseSettings()
    @add_arg_table arg_parser begin
        "--experiment-name", "-n"
            help = "name of output experiment"
            required = false
            default = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
        "--config-name", "-c"
            help = "name of config in configs/ without extension such as default"
            required = false
            default = "default"
    end
    args = parse_args(arg_parser)

    config_path = joinpath("configs", "$(args["config-name"]).yml")

    # Create experiment directory
    experiment_name = "TRAIN-" * args["experiment-name"]
    experiment_path = mkpath("outputs/$(experiment_name)/")
    mkpath("outputs/$(experiment_name)/plots/")
    
    # Load and save config
    config = YAML.load_file(config_path)
    cp(config_path, joinpath(experiment_path, "config.yml"), force=true)

    train(config, experiment_path)
end

main()