module GeneticAlgorithm
include("./environment/neural_network.jl")
include("./environment/objects/individual.jl")
import StatsBase, Random, Distributions
import .NeuralNetwork, .IndividualModule
export evolve_population

function elitism(population, pop_fitnesses, elite_frac, method = "top_selection", kwargs... = nothing)
    # Holder for surviving pops and number of pop that will survive
    surviving_pop = []
    elite_num = round(Int, elite_frac * length(population))

    if method == "top_selection"
        # Top n elitism
        if elite_num < 1 throw(ArgumentError("elite_frac too small")) end
        permvec = sortperm(pop_fitnesses, rev=true)
        sorted_pop = population[permvec]
        surviving_pop = sorted_pop[1 : elite_num]

    elseif method == "k_tournament"
        if kwargs === nothing
            throw(ArgumentError("Method needed kwargs but kwargs were not given"))
        end
        # Unpack kwargs, if no k then default to 5
        kwargs = Dict(kwargs)
        if haskey(kwargs, "k")
            k_quantity = kwargs["k"]
        else
            k_quantity = 5
        end
        for survivor in 1:elite_num
            # Find a set of individuals and their coresponding weights
            tournament = StatsBase.sample(population, k_quantity, replace=false)
            weights = []
            tournament_fitnesses = [ind.fitness + 1 for ind in tournament]
            weights = StatsBase.ProbabilityWeights(tournament_fitnesses)
            # Select parent from tournament based on weight
            push!(surviving_pop, StatsBase.sample(tournament, weights, 1, replace=false)[1])
        end
    else
        throw(ArgumentError("Invalid method selected"))
    end
    return surviving_pop
end

function parent_selection(population, pop_fitnesses_as_weights, method = "roultette", kwargs... = nothing)

    if method == "roultette"                # Roulette selection | Destructive
        return StatsBase.sample(population, pop_fitnesses_as_weights, 2, replace=false)

    elseif method == "k_tournament"         # Tournament selection
        if kwargs === nothing
            throw(ArgumentError("Method needed kwargs but kwargs were not given"))
        end

        # Unpack kwargs, if no k then default to 5
        kwargs = Dict(kwargs)
        if haskey(kwargs, "k")
            k_quantity = kwargs["k"]
        else
            k_quantity = 5
        end

        # Perform two tournaments of size k to get two parents
        parents = []
        for parentNum in 1:2

            # Find a set of individuals and their coresponding weights
            tournament = StatsBase.sample(population, k_quantity, replace=true)
            weights = []
            tournament_fitnesses = [ind.fitness + 1 for ind in tournament]
            weights = StatsBase.ProbabilityWeights(tournament_fitnesses)

            # Select parent from tournament based on weight
            push!(parents,StatsBase.sample(tournament, weights, 1, replace=false)[1])
        end
        return parents
    else
        throw(ArgumentError("Invalid method selected"))
    end
end

function crossover(parents, method = "one_point_crossover", kwargs... = nothing)
    # Extract chromosomes from partents
    chrom_a = parents[1].chromosome
    chrom_b = parents[2].chromosome
    
    # Make sure chromosome lengths are the same size
    if length(chrom_a) != length(chrom_b)
        throw(ArgumentError("Parents have non-equal chromosome lengths"))
    end

    if method == "one_point_crossover"
        # Single point crossover
        split_point = Random.rand(1 : length(chrom_a) - 1)

        first_part = chrom_a[1 : split_point]
        second_part = chrom_b[split_point + 1 : end]

        new_chrom = vcat(first_part, second_part)
        return new_chrom

    elseif method == "two_point_crossover"
        # Two point crossover

        # Create two random points and make sure they arent equal
        split_point_1 = Random.rand(1 : length(chrom_a) - 1)
        split_point_2 = Random.rand(1 : length(chrom_a) - 1)
        while split_point_1 == split_point_2
            split_point_2 = Random.rand(1 : length(chrom_a) - 1)
        end
        smaller_split_point = min(split_point_1,split_point_2)
        larger_split_point = max(split_point_1, split_point_2)

        # Split chromosomes
        first_part_a = chrom_a[1 : smaller_split_point]
        second_part_a = chrom_a[smaller_split_point + 1 : larger_split_point]
        third_part_a = chrom_a[larger_split_point + 1 : end]
        first_part_b = chrom_b[1 : smaller_split_point]
        second_part_b = chrom_b[smaller_split_point + 1 : larger_split_point]
        third_part_b = chrom_b[larger_split_point + 1 : end]

        # 50/50 shot a donates middle, b donates middle
        coin = Random.rand()
        if coin < 0.5
            # Heads = a donates middle
            partial_chrom = vcat(first_part_b, second_part_a)
            new_chrom = vcat(partial_chrom, third_part_b)
        else
            # Tails = b donates middle
            partial_chrom = vcat(first_part_a, second_part_b)
            new_chrom = vcat(partial_chrom, third_part_a)
        end
        return new_chrom

    elseif method == "uniform_crossover"
        # Uniform Crossover
        new_chrom = zeros(length(chrom_a))

        # Create array that will decide whose gene to pick for each slot available
        decision_array = zeros(length(chrom_a))
        Random.rand!(decision_array)

        # 50/50 shot a or b will be chosen if an index is > 0.5
        coin = Random.rand()
        if coin < 0.5
            # Heads = a is all >0.5 values
            for i in eachindex(new_chrom)
                if decision_array[i] > 0.5
                    new_chrom[i] = chrom_a[i]
                else
                    new_chrom[i] = chrom_b[i]
                end
            end
        else
            # tails = b is all >0.5 values
            for i in eachindex(new_chrom)
                if decision_array[i] > 0.5
                    new_chrom[i] = chrom_b[i]
                else
                    new_chrom[i] = chrom_a[i]
                end
            end
        end
        return new_chrom 
    else
        throw(ArgumentError("Invalid method selected"))
    end
end

function mutate(child_chrom, mutation_prob_max, mutation_frac_change_max)
    # Single gene mutation
    mutation_prob = Random.rand(Distributions.Uniform(0., 1.))
    if mutation_prob < mutation_prob_max
        # Select random gene
        gene_to_mutate = Random.rand(1 : length(child_chrom))
        # Select amount to mutate
        mutation_frac_change = Random.rand(Distributions.Uniform(-mutation_frac_change_max, mutation_frac_change_max))
        amt_to_change = child_chrom[gene_to_mutate] * mutation_frac_change
        # Mutate
        child_chrom[gene_to_mutate] += amt_to_change
    end
    return child_chrom
end

function evolve_population(ga_config, population, nn_size, 
                            individual_config, neural_network_config, next_obj_id)
    # Given fitness of individuals evolve using GA
    pop_size = length(population)
    pop_fitnesses = [ind.fitness + 1 for ind in population]
    pop_fitnesses_as_weights = StatsBase.ProbabilityWeights(pop_fitnesses)

    # 1. Elitism: Copy a certain percent of current pop to next pop
    elite_pop = elitism(population, pop_fitnesses, ga_config["elite_frac"], ga_config["elite_method"], ga_config)
    remaining_children_req = pop_size - length(elite_pop)

    # 2. Parent Selection, Crossover and Mutation
    children = []
    for _ = 1 : remaining_children_req
        parents = parent_selection(population, pop_fitnesses_as_weights, ga_config["parent_method"], ga_config)
        child_chrom = crossover(parents, ga_config["crossover_method"], ga_config)
        child_chrom = mutate(child_chrom, ga_config["mutation_prob_max"], ga_config["mutation_frac_change_max"])
        child = IndividualModule.create_new_individual(next_obj_id, child_chrom, nn_size, individual_config, neural_network_config)
        push!(children, child)
    end

    new_pop = vcat(elite_pop, children)
    return new_pop
end

end