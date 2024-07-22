# concurrent-ga-julia

Evolves individuals to better collect food in a virtual environment using a genetic algorithm and neural network movement models

### Entrypoints:
- train.jl: For training a GA: `julia --threads auto train.jl --help`
- evaluate.jl: For evaluating a trained GA: `julia --threads auto evaluate.jl --help`
    - Example: `julia --threads auto evaluate.jl -t <train-dir-name>`

### Example Running
`julia --threads auto train.jl -n example -c example_train`
`julia --threads auto evaulate.jl -n example -t TRAIN-example -o example_eval`