module NeuralNetwork
import Flux, StatsBase
export make_model, get_action, calc_model_dims

function calc_model_dims(individual_config, nn_config)
    input_size = length(individual_config["inputs"])
    hidden_size = nn_config["num_hidden_layer_nodes"]
    output_size = individual_config["num_actions"]

    first_layer_length = input_size * hidden_size + hidden_size
    second_layer_length = hidden_size * output_size + output_size
    num_weights = first_layer_length + second_layer_length

    return (input_size, hidden_size, output_size), num_weights
end

function set_weight_and_bias_genes(layer_genes, in_size, out_size, model, model_weight_indx)
    weight_genes = layer_genes[1 : in_size * out_size]
    bias_genes = layer_genes[length(weight_genes) + 1 : end]
    weight_genes = reshape(weight_genes, (out_size, in_size))

    model[model_weight_indx].weight .= weight_genes
    model[model_weight_indx].bias .= bias_genes

    return model
end

function update_model(chromosome, model, nn_size)
    num_input, num_hidden_nodes, num_output = nn_size[1], nn_size[2], nn_size[3]

    first_layer_length = num_input * num_hidden_nodes + num_hidden_nodes

    first_layer_genes = chromosome[1 : first_layer_length]
    second_layer_genes = chromosome[first_layer_length + 1 : end]

    model = set_weight_and_bias_genes(first_layer_genes, num_input, num_hidden_nodes, model, 1)
    model = set_weight_and_bias_genes(second_layer_genes, num_hidden_nodes, num_output, model, 2)

    return model
end

function get_act_func(hidden_layer_act_func)
    if hidden_layer_act_func == "tanh"
        return Flux.tanh_fast
    elseif hidden_layer_act_func == "sigmoid"
        return Flux.sigmoid_fast
    else
        @assert hidden_layer_act_func == "relu"
        return Flux.relu
    end
end

function make_model(nn_size, chromosome, hidden_layer_act_func)
    num_input, num_hidden_nodes, num_output = nn_size[1], nn_size[2], nn_size[3]
    model = Flux.Chain(Flux.Dense(nn_size[1] => nn_size[2], get_act_func(hidden_layer_act_func)),
                      Flux.Dense(nn_size[2] => nn_size[3]))
    model = update_model(chromosome, model, nn_size)
    return model
end

function get_action(model, input, action_sampling)
    out = model(input)
    # See if moving on x or y axis
    if action_sampling == "prob"
        axis = StatsBase.sample(["x", "y"], StatsBase.ProbabilityWeights([abs(x) for x in out]))
    else
        axis = ["x", "y"][findmax([abs(x) for x in out])[2]]
    end
    # Select action
    if axis == "x"
        if out[1] >= 0
            action = "right"
        else
            action = "left"
        end
    else
        if out[2] >= 0
            action = "down"
        else
            action = "up"
        end
    end
    return action
end

end