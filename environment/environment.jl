module EnvironmentModule
import StatsBase
using Random
export Environment, create_environment, render_environment

function create_environment(environment_config, population)
    num_rows, num_cols = environment_config["num_rows"], environment_config["num_rows"]
    fraction_food = environment_config["fraction_food"]

    # Make empty cells
    empty_cell_positions = Set()
    cells = Array{Cell}(undef, num_rows, num_cols)
    for r = 1 : num_rows
        for c = 1 : num_cols
            cells[r, c] = Cell(position=(c, r))
            push!(empty_cell_positions, (c, r))
        end
    end

    # Add individuals
    selected_cells = StatsBase.sample(cells, length(population), replace=false)
    for i = 1 : length(selected_cells)
        cell = selected_cells[i]
        ind = population[i]
        ind.position = cell.position
        cell.objects[ind.id] = ind
        delete!(empty_cell_positions, cell.position)
    end

    # Add food (as a property of the environment)
    amt_food_left = spawn_food(cells, empty_cell_positions, round(Int, fraction_food * length(population)))

    environment = Environment(num_rows=num_rows, num_cols=num_cols, cells=cells,
                              amt_food_left=amt_food_left, empty_cell_positions=empty_cell_positions)
    return environment
end

function spawn_food(cells, empty_cell_positions, amt)
    added_food = 0
    selected_cells = StatsBase.sample([p for p in empty_cell_positions], amt, replace=false)
    for cell_pos in selected_cells
        cell = cells[cell_pos[2], cell_pos[1]]
        @assert cell.has_food == false
        @assert length(cell.objects) == 0
        cell.has_food = true
        delete!(empty_cell_positions, cell_pos)
        added_food += 1
    end
    return added_food
end

Base.@kwdef mutable struct Cell
    objects = Dict()
    lk = ReentrantLock()
    position
    has_food = false
end

Base.@kwdef mutable struct Environment
    num_rows
    num_cols
    cells
    amt_food_left
    lk = ReentrantLock()
    empty_cell_positions
end

end