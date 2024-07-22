module IndividualModule
include("../neural_network.jl")
import .NeuralNetwork
import Flux
export create_new_individual, create_initial_population, get_action, reset_population_before_episode

function create_initial_population(population_config, chromosome_length, nn_size, 
                                    individual_config, neural_network_config, 
                                    next_obj_id, chromosomes=nothing)
    # Create population of new individuals given config
    # This doesnt need to "spawn" the individuals in the environment yet, just create things such 
    # as their genes
    pop = []
    for i = 1 : population_config["pop_size"]
        if chromosomes === nothing
            chromosome = Flux.kaiming_normal(chromosome_length)
        else
            chromosome = chromosomes[i]
        end
        ind = create_new_individual(next_obj_id, chromosome, nn_size, individual_config, neural_network_config)
        push!(pop, ind)
    end
    return pop
end

function get_action(ind, input)
    return NeuralNetwork.get_action(ind.movement_model, input, ind.action_sampling)
end

function reset_population_before_episode(population)
    for individual in population
        individual.fitness = 0
    end
end

function create_new_individual(next_obj_id, chromosome, nn_size, individual_config, neural_network_config)
    movement_model = NeuralNetwork.make_model(nn_size, chromosome, neural_network_config["hidden_layer_act_func"])
    individual = Individual(id=next_obj_id.next_id, chromosome=chromosome,
                            movement_model=movement_model, sensor_radius=individual_config["sensor_radius"],
                            inputs=individual_config["inputs"], action_sampling=individual_config["action_sampling"])
    next_obj_id.next_id += 1
    return individual
end

Base.@kwdef mutable struct Individual
    id
    fitness = 0  # If just food in an enviroment, then this is ++ for each food collected
    chromosome
    movement_model  # neural network used for movement in game engine, reset every evolution
    inputs
    sensor_radius
    position = (1, 1)
    action_sampling
end

end