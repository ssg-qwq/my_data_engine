from utils.tools import dict_to_namespace

# <|SOC|>[<|SOM|> tokens <|EOM|> <|SOM|> tokens <|EOM|> ...]<|EOC|>
# <|SOD|> tokens <|EOD|>
special_token_dict = {
    "taste": 65520, # <|taste|>
    "touch": 65521,  # <|touch|>
    "heatmap": 65522, # <|heatmap|>
    "pc_vision": 65523, # <|pc_vision|>
    "video": 65524,  # <|video|>
    "audio": 65525,  # <|audio|>
    "motion": 65526,  # <|motion|>
    "start_of_condition_block": 65527,  # <|SOC|>
    "end_of_condition_block": 65528,  # <|EOC|>
    "start_of_dynamic_block": 65529,  # <|SOD|>
    "end_of_dynamic_block": 65530,  # <|EOD|>
    "system": 65531,  # <|system|>
    "conversation": 65532,  # <|conversation|>
    "think": 65533,  # <|think|>
    "response": 65534,  # <|response|>
    "end_of_modal": 65535,  # <|EOM|>
}

special_token_config = dict_to_namespace(special_token_dict)
