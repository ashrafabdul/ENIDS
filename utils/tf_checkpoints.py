from tensorflow.python import pywrap_tensorflow

BEST_CHECKPOINT_MTL = '/home/aabdul/projects/enids/data/NSL-KDD/master/model/best_checkpoint/mtl_tt_ac_at/29096'

def get_model_parameters(file_name, tensor_names):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    keys = []
    for key in var_to_shape_map:
        print(reader.get_tensor(key))
        keys.append(key)
    print(keys)


if __name__ == "__main__":

    tensor_names = ['hl_1/W','hl_1/b','hl_2/W','hl_2/b','hl_3/W','hl_3/b','hl_4/W','hl_4/b','hl_5/W','hl_5/b','hl_6/W','hl_6/b','ac_output/W','ac_output/b']
    get_model_parameters(BEST_CHECKPOINT_MTL, tensor_names)