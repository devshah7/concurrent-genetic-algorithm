module Utils
using OrderedCollections
export NextObjId, override_config

mutable struct NextObjId
    next_id
end

function override_config(base_config, override_config)
    for (section_key, section_dic) in override_config
        for (key, value) in section_dic
            base_config[section_key][key] = value
        end
    end
end

end