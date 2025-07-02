from utils.tools import dict_to_namespace

# 1. condition special token 的对应位置mask应该为0
special_token_dict = {
    "heatmap_condition": [65528, 14195],  # <|SOM|>热
    "image_condition": [65528, 16523],  # <|SOM|>视
    "audio_condition": [65528, 11065],  # <|SOM|>听
    "touch_condition": [65528, 16535],  # <|SOM|>触
    "taste_condition": [65528, 11110],  # <|SOM|>味
    "pc_vision_condition": [65528, 12019],  # <|SOM|>屏
    "image_dynamic": [65525],  # <|image_dynamic|>
    "audio_dynamic": [65526],  # <|audio_dynamic|>
    "motion_dynamic": [65527],  # <|motion_dynamic|>
    "text_dynamic": [65534],  # <|response|>
    "end_of_modal": [65529],  # <|EOM|>
    "text_condition": [65530, 65532],  # <|start|><|conversation|>
    "conversation": [65530, 65532],  # <|start|><|conversation|>
    "think": [65530, 65533],  # <|start|><|think|>
    "think_condition": [65530, 65533],  # <|start|><|think|>
    "response": [65530, 65534],  # <|start|><|response|>
    "response_condition": [65530, 65532],  # <|start|><|text_condition|>
    "end_of_text":[65535], # <|over|>
}


special_token_config = dict_to_namespace(special_token_dict)
