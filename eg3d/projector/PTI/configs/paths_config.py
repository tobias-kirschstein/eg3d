## Pretrained models paths
from eg3d.env import EG3D_MODELS_PATH

e4e = './pretrained_models/e4e_ffhq_encode.pt'
stylegan2_ada_ffhq = '../pretrained_models/ffhq.pkl'
# eg3d_ffhq_pkl = '../../networks/ffhq512-128.pkl'
# eg3d_ffhq_pkl = '../../../../models/ffhq512-128.pkl'
eg3d_ffhq_pkl = f"{EG3D_MODELS_PATH}/ffhq512-128.pkl"
eg3d_ffhq_rebalanced_pkl = f"{EG3D_MODELS_PATH}/ffhqrebalanced512-128.pkl"
# eg3d_ffhq_pth = '../../networks/ffhq512-128.pth'
eg3d_ffhq_pth = '../../../../models/ffhq512-128.pth'
style_clip_pretrained_mappers = ''
ir_se50 = './pretrained_models/model_ir_se50.pth'
dlib = './pretrained_models/align.dat'

## Dirs for output files
checkpoints_dir = './checkpoints'
embedding_base_dir = './embeddings'
styleclip_output_dir = './StyleCLIP_results'
experiments_output_dir = './output'

## Input info
### Input dir, where the images reside
input_data_path = '../../projector_test_data/00025.png'
### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = '00025'

input_c_path = '../../projector_test_data'

## Keywords
pti_results_keyword = 'PTI'
e4e_results_keyword = 'e4e'
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

## Edit directions
interfacegan_age = 'editings/interfacegan_directions/age.pt'
interfacegan_smile = 'editings/interfacegan_directions/smile.pt'
interfacegan_rotation = 'editings/interfacegan_directions/rotation.pt'
ffhq_pca = 'editings/ganspace_pca/ffhq_pca.pt'
