# Deals running episode, ie interactions between environment and objects
# Does some things concurrently for speed-up

# - movememnt is achieved with a neural network. The genes are the weights. 
#   Input is various objects the individual can see. Output is direction of movement (and if to move at all)
module GameEngine
include("./objects/individual.jl")
include("./environment.jl")
import .Threads, .IndividualModule, .EnvironmentModule
using Random
export run_episode

function gather_input_for_individual(individual, environment, missing_input_position_padding)
    ind_inputs = individual.inputs
    r = individual.sensor_radius

    ind_x, ind_y = individual.position[1], individual.position[2]
    num_rows, num_cols = environment.num_rows, environment.num_cols
    cells = environment.cells

    # Gather cells in radius (including the cell this individual is standing on)
    # Get inputs while doing this
    closest_food_rel_pos, closest_food_dist, is_food = (r, r), Inf, false
    closest_ind_rel_pos, closest_ind_dist, is_ind = (r, r), Inf, false
    closest_perim_rel_pos, closest_perim_dist, is_perim = (r, r), Inf, false
    for y = max(ind_y - r, 1) : min(ind_y + r, num_rows)
        for x = max(ind_x - r, 1) : min(ind_x + r, num_cols)
            rel_x = x - ind_x
            rel_y = y - ind_y
            # Check if within sensor circle
            if rel_x ^ 2 + rel_y ^ 2 <= r ^ 2
                dist_to_cell = abs(rel_x) + abs(rel_y)
                cell = cells[y, x]
                # Check food
                if cell.has_food
                    if dist_to_cell < closest_food_dist
                        closest_food_dist = dist_to_cell
                        closest_food_rel_pos = (rel_x, rel_y)
                        is_food = true
                    end
                end
                # Check individual
                # IMPORTANT NOTE: Right now the only possible object is an individual, so this just checks length
                # Also it is possible for another individual to be in the same cell as this individual
                # so check if it is the same cell that the length is at least 2, otherwise length at least 1
                same_cell = x == ind_x && y == ind_y
                if (same_cell && length(cell.objects) >= 2) || (same_cell == false && length(cell.objects) >= 1)
                    if dist_to_cell < closest_ind_dist
                        closest_ind_dist = dist_to_cell
                        closest_ind_rel_pos = (rel_x, rel_y)
                        is_ind = true
                    end
                end
                # Check perimeter
                if x == 1 || x == num_cols || y == 1 || y == num_rows
                    if dist_to_cell < closest_perim_dist
                        closest_perim_dist = dist_to_cell
                        closest_perim_rel_pos = (rel_x, rel_y)
                        is_perim = true
                    end
                end
            end
        end 
    end

    # Scale inputs between -1 and 1
    scale(x, ma) = x / ma
    function scale_input_position(rel_pos)
        scaled_pos = [0., 0.]
        for (pos_indx, pos_max) in [(1, r), (2, r)]
            if rel_pos[pos_indx] == pos_max
                scaled = missing_input_position_padding
            else
                scaled = scale(rel_pos[pos_indx], pos_max)
            end
            scaled_pos[pos_indx] = scaled
        end
        return Tuple(scaled_pos)
    end

    nn_inputs = []
    if "closest_food_x" in ind_inputs && "closest_food_y" in ind_inputs
        closest_food = scale_input_position(closest_food_rel_pos)
        push!(nn_inputs, closest_food[1])
        push!(nn_inputs, closest_food[2])
    end
    if "is_food" in ind_inputs
        push!(nn_inputs, convert(Float16, is_food))
    end
    if "closest_individual_x" in ind_inputs && "closest_individual_y" in ind_inputs
        closest_ind = scale_input_position(closest_ind_rel_pos)
        push!(nn_inputs, closest_ind[1])
        push!(nn_inputs, closest_ind[2])
    end
    if "is_individual" in ind_inputs
        push!(nn_inputs, convert(Float16, is_ind))
    end
    if "closest_perimeter_x" in ind_inputs && "closest_perimeter_y" in ind_inputs
        closest_perim = scale_input_position(closest_perim_rel_pos)
        push!(nn_inputs, closest_perim[1])
        push!(nn_inputs, closest_perim[2])
    end
    if "is_perimeter" in ind_inputs
        push!(nn_inputs, convert(Float16, is_perim))
    end
    return nn_inputs
end

function move_individual(individual, action, environment)
    move_x, move_y = 0, 0
    if action == "left"
        move_x -= 1
    elseif action == "right"
        move_x += 1
    elseif action == "up"
        move_y -= 1
    elseif action == "down"
        move_y += 1
    end

    old_x, old_y = individual.position[1], individual.position[2]
    new_x, new_y = old_x + move_x, old_y + move_y

    # Cannot move out-of-bounds
    new_x = min(max(new_x, 1), environment.num_cols)
    new_y = min(max(new_y, 1), environment.num_rows)

    # Update position of ind
    individual.position = (new_x, new_y)

    # Remove from old cell, move to new cell
    old_cell = environment.cells[old_y, old_x]
    new_cell = environment.cells[new_y, new_x]

    if new_x != old_x || new_y != old_y
        lock(old_cell.lk) do
            delete!(old_cell.objects, individual.id)
            # Add to freed cells IF no other ind in this cell
            if length(old_cell.objects) == 0
                lock(environment.lk) do 
                    push!(environment.empty_cell_positions, (old_x, old_y))
                    @assert old_cell.has_food == false
                end
            end
        end
        lock(new_cell.lk) do
            new_cell.objects[individual.id] = individual
            # If food here then remove and add to ind fitness
            if new_cell.has_food == true
                lock(environment.lk) do
                    environment.amt_food_left -= 1
                end
                new_cell.has_food = false
                individual.fitness += 1
            end
            # Remove from freed cells, unless there was an ind here
            if length(new_cell.objects) == 1
                lock(environment.lk) do
                    delete!(environment.empty_cell_positions, (new_x, new_y))
                end
            end
        end
    end
end

function do_individual_action(individual, environment, missing_input_position_padding)
    # Collect observations (ie input for neural network model)
    input = gather_input_for_individual(individual, environment, missing_input_position_padding)
    # Feed into model to get action
    action = IndividualModule.get_action(individual, input)
    # Update environment given action
    move_individual(individual, action, environment)
end

function get_next_food_spawn_step(food_respawn_range, will_respawn_food, curr_step)
    if will_respawn_food
        return rand(food_respawn_range[1] : food_respawn_range[2]) + curr_step
    end
    return Inf
end

function respawn_food(environment, amt)
    if amt > 0
        added_food = EnvironmentModule.spawn_food(environment.cells, environment.empty_cell_positions, amt)
        environment.amt_food_left += added_food
    end
end

function check_correctness(environment)
    empty = Set()
    num_food = 0
    for r = 1 : environment.num_rows
        for c = 1 : environment.num_cols
            cell = environment.cells[r, c]
            if cell.has_food
                num_food += 1
                @assert length(cell.objects) == 0
            end
            if cell.has_food == false && length(cell.objects) == 0
                push!(empty, cell.position)
            end
        end
    end
    @assert issetequal(empty, environment.empty_cell_positions)
    @assert num_food == environment.amt_food_left
end

function run_episode(episode_config, environment_config, population, missing_input_position_padding, render=false)
    frames = []
    
    # Create new enviroment
    environment = EnvironmentModule.create_environment(environment_config, population)

    # Shuffle individuals
    shuffle!(population)

    food_respawn_range = episode_config["food_respawn_range"]
    will_respawn_food = food_respawn_range !== nothing
    next_spawn_step = get_next_food_spawn_step(food_respawn_range, will_respawn_food, 1)
    amt_food_at_start = environment.amt_food_left

    # Run episode steps
    for step = 1 : episode_config["max_num_steps"]
        # Concurrently move individuals (use move_individual above)
        Threads.@threads for individual in population
            do_individual_action(individual, environment, missing_input_position_padding)
        end
        if render
            push!(frames, deepcopy(environment))
        end
        # Respawn food
        if step == next_spawn_step
            respawn_food(environment, amt_food_at_start - environment.amt_food_left)
            next_spawn_step = get_next_food_spawn_step(food_respawn_range, will_respawn_food, step)
        end

        # check_correctness(environment)

        # Break if no more food left in environment
        if will_respawn_food == false && environment.amt_food_left == 0
            break
        end
    end
    return frames
    
end

end