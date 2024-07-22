module RenderEnvironment
include("./environment/environment.jl")
using Luxor
import .EnvironmentModule
export render_environment

fps = 3
cell_length = 25
colors_array = ["orange", "red", "purple", "blue"]

function backdrop(scene, framenumber)
    background("black")
end

function frame(scene, framenumber)
    setfont("Georgia Bold", cell_length - 8)
    origin()
    arr = scene.opts["arrays"][framenumber]
    tiles = scene.opts["tiles"]
    num_cols = scene.opts["num_cols"]
    for (pos, n) in tiles
        sethue("gray")
        box(pos, tiles.tilewidth, tiles.tileheight, action=:stroke)
        sethue("white")
        x = ((n-1)%num_cols) + 1
        y = (floor(Int, (n-1)/num_cols)) + 1
        cell = arr.cells[y, x]
        num_objs = length(cell.objects)
        if num_objs > 0
            if num_objs <= length(colors_array)
                color = colors_array[num_objs]
            else
                color = colors_array[end]
            end
            sethue(color)
            ellipse(pos, tiles.tilewidth / 2, tiles.tileheight / 2 , action=:fill) 
        elseif cell.has_food
            sethue("green")
            ellipse(pos, tiles.tilewidth / 4, tiles.tileheight / 4, action=:fill)
        end
    end
end

function render_environment(environment_config, environment_frames, render_dir)
    num_rows = environment_config["num_rows"]
    num_cols = num_rows

    window_height = cell_length * num_rows
    window_width = window_height

    render = Movie(window_width, window_height, "render")
    tiles = Tiler(window_width, window_height, num_rows, num_cols, margin=0)

    num_frames = length(environment_frames)

    animate(render, [
        Scene(render, backdrop, 1:num_frames),
        Scene(render, frame, 1:num_frames,
            easingfunction=lineartween,
            optarg=Dict("arrays" => environment_frames, "tiles" => tiles, "num_cols" => num_cols))
        ],
        creategif=true,
        framerate=fps,
        pathname=joinpath(render_dir, "render.gif"))
end

end