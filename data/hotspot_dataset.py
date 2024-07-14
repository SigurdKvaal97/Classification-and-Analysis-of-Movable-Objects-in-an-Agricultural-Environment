import torch.utils.data as tdata
import collections
import torch
import numpy as np
import h5py
from joblib import Parallel, delayed
from PIL import Image
from utils import util



class VideoInteractions(tdata.Dataset):

    def __init__(self, root, split, max_len, sample_rate):
        self.root = root
        self.split = split
        self.max_len = max_len
        self.sample_rate = sample_rate

        if self.max_len==-1:
            self.max_len = 32
            print ('Max length not chosen. Setting max length to:', self.max_len)
        
        self.clip_transform = util.clip_transform(self.split, self.max_len)
        self.img_transform = util.default_transform(self.split)
        self.pair_transform = util.PairedTransform(self.split)
    
    # Function to load each frame in entry['frames']
    # return: PIL image for frame
    def load_frame(self, frame):
        pass

    # Function to select the positive (and optional negative) inactive image for L_{ant}
    # return: positive (3,224,224), negative (3,224,224)
    def select_inactive_instances(self, entry):
        pass

    # Weighted sampler for class imbalance
    def data_sampler(self):
        counts = collections.Counter([entry['verb'] for entry in self.data])
        icounts = {k:sum([counts[v] for v in counts])/counts[k] for k in counts}
        icounts = {k:min(icounts[k], 100) for k in icounts}
        weights = [icounts[entry['verb']] for entry in self.data]
        weights = np.array(weights)
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        return sampler

    # sample a randomly placed window of self.max_len frames from the video clip
    def sample(self, clip):

        if len(clip)<=self.max_len:
            return clip

        if self.split=='train':
            st = np.random.randint(0, len(clip)-self.max_len)
        elif self.split=='val':
            st = len(clip)//2 - self.max_len//2
        clip = clip[st:st+self.max_len]

        return clip

    def __getitem__(self, index):
        # intrinsics for depth
        K_ir= np.array([[935.29634243,   0,         630.61455831], 
            [  0,         935.29634243, 367.4203064 ],
            [  0,           0,           1,        ]])
        dist_ir=np.array([[-1.09210660e-01,  1.03981748e+00,  9.86420068e-04, -2.83077025e-03, -2.73889538e+00]])
        K_tobii = np.array([[915.77932432,   0,         962.20888519],
                [  0,         915.77932432, 509.89242024],
                [  0,           0,           1,        ]])
        dist_tobii=np.array([[-0.0674579,   0.1682862,  -0.00355576,  0.00125199, -0.19291438]])
        T = np.array([[ 9.99052211e-01, -1.01866331e-02, -4.23191814e-02,  3.29833519e+01],
            [ 1.81602585e-02,  9.81118194e-01,  1.92554653e-01, -3.21970445e+00],
            [ 3.95586352e-02, -1.93140679e-01,  9.80373292e-01, -5.50198898e-01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

        entry = self.data[index]

        #--------------------------------------------------------------------------#
        gaze_points = []

        gaze_frames= []
        
        depth_frames = []

        #from EPICInteractions(VideoInteractions)
        gaze_annot = self.gaze
        gaze_path = "/home/filip_lund_andersen/gaze"
        
        # sample frames and load/transform
        frames = self.sample(entry['frames'])
        length = len(frames)
        
        for frame in frames:
            
            v_id, f_id = frame[0], frame[1]
            fg_id = f_id + 1
            

            # depth start   ----------------------------------------------------
            
            from data.extract_depth_im import depth_to_tobii
            import cv2 

            if v_id == "M01_01":
                start_frame_tobii = 169
                start_frame_intel = 387
            elif v_id == "M01_04":
                start_frame_tobii = 111
                start_frame_intel = 419
            elif v_id == "M02_01":
                start_frame_tobii = 72
                start_frame_intel = 404
            elif v_id == "M02_02":
                start_frame_tobii = 151
                start_frame_intel = 491
            elif v_id == "M02_03":
                start_frame_tobii = 164
                start_frame_intel = 559
            elif v_id == "M02_04":
                start_frame_tobii = 81
                start_frame_intel = 488
            elif v_id == "M02_05":
                start_frame_tobii = 78
                start_frame_intel = 375
            elif v_id == "M02_06":
                start_frame_tobii = 95
                start_frame_intel = 434
            elif v_id == "M02_07":
                start_frame_tobii = 114
                start_frame_intel = 413
            elif v_id == "M02_08":
                start_frame_tobii = 93
                start_frame_intel = 392
            elif v_id == "M04_01":
                start_frame_tobii = 160
                start_frame_intel = 408
            elif v_id == "M04_02":
                start_frame_tobii = 31
                start_frame_intel = 219
            else:
                start_frame_intel = 0
                start_frame_tobii = 0


            depth_base_path = "/home/filip_lund_andersen/depth_npy"
            frame_base_path = f"/home/filip_lund_andersen/MasterData/frames_rgb_flow/rgb/train/{v_id[:3]}/{v_id}"
            
            f_id_intel_adjust = int(start_frame_intel + ((fg_id - start_frame_tobii) * (30/25)))

            num_zero = 6 - len(str(f_id_intel_adjust))

            zeros = "0"*num_zero

            if str(v_id) == "M01_01":
                path_depth = depth_base_path + "/" + str(v_id) + "/" + "Depth_" + str(zeros) + str(f_id_intel_adjust) + ".npy"

            elif str(v_id) == "M04_01":
                path_depth = depth_base_path + "/" + str(v_id) + "/M04_1" + "/" + "Depth_" + str(zeros) + str(f_id_intel_adjust) + ".npy"
            elif str(v_id) == "M04_02":
                path_depth = depth_base_path + "/" + str(v_id) + "/M04_3" + "/" + "Depth_" + str(zeros) + str(f_id_intel_adjust) + ".npy"

            else:
                path_depth = depth_base_path + "/" + str(v_id) + "/" + str(v_id) + "/" + "Depth_" + str(zeros) + str(f_id_intel_adjust) + ".npy"
            

            num_zero = 10- len(str(f_id))

            zeros = "0"*num_zero
            path_tobii = frame_base_path + "/" + "frame_" + str(zeros) + str(f_id) + ".jpg"
            
            
            list_of_depth_M = ["M01_01", "M01_04", "M02_01", "M02_02", "M02_03", "M02_04", "M02_05", "M02_06",
                                "M02_07", "M02_08", "M04_01"]
            
            if v_id in list_of_depth_M:
                try:
                    img_d = np.load(path_depth) 
                    img_tobii = cv2.imread(path_tobii)
                    
                    depth_frame = depth_to_tobii(img_tobii, img_d, K_tobii, K_ir, dist_tobii, dist_ir, T)
                   
                    

                    # Normalize depth values to the range [0, 255]
                    normalized_depth_frame = ((depth_frame - depth_frame.min()) / (depth_frame.max() - depth_frame.min()) * 255).astype(np.uint8)

                    # Create a PIL image from the normalized array
                    depth_frame = Image.fromarray(normalized_depth_frame)
                except FileNotFoundError:
                    depth_frame = np.zeros((1080, 1920), dtype=np.uint8)

                    # Stack the depth frame into three channels to create an RGB image
                    depth_frame = np.stack((depth_frame,) * 3, axis=-1)
                    depth_frame = Image.fromarray(depth_frame)
            

            else:
            
                depth_frame = np.zeros((1080, 1920), dtype=np.uint8)

                # Stack the depth frame into three channels to create an RGB image
                depth_frame = np.stack((depth_frame,) * 3, axis=-1)
                depth_frame = Image.fromarray(depth_frame)
            
            
            
            gaze_width, gaze_height = (1080, 1920)
            
            depth_frames.append(depth_frame)
            
            
            # depth slutt   ----------------------------------------------
            # gaze gistory start  ----------------------------------------------
            
            #included if using gaze history
            
            if v_id[:3] != "M05":
                
                num_space = 10- len(str(fg_id))
                space = " "*num_space
                tmp_gaze_path = gaze_path + "/" + str(v_id) + "/gaze_"+ str(space) + str(fg_id) + ".png"
                tmp_gaze_path= f"{tmp_gaze_path}"
                gaze_img = Image.open(tmp_gaze_path).convert("RGB")
            else:
                image_size = (1080, 1920, 3)   # (width, height)

                # Create a numpy array of zeros
                zero_array = np.zeros(image_size, dtype=np.uint8)

                # Create an image from the numpy array
                gaze_img = Image.fromarray(zero_array)

            
            gaze_frames.append(gaze_img)
            
            # gaze history end   ----------------------------------------------
            
            gaze_point_x = gaze_annot[v_id][str(f_id)]["gaze_x"]
            gaze_point_y = gaze_annot[v_id][str(f_id)]["gaze_y"]
            gaze_points.append([gaze_point_x, gaze_point_y])

        
        gaze_points = np.array(gaze_points, dtype=np.float32)

        
        # Pad gaze points to match max_len
        padded_gaze_points = np.zeros((self.max_len, 2))
        padded_gaze_points[:len(gaze_points), :] = gaze_points

        padded_gaze_points = np.array(padded_gaze_points, dtype=np.float32)
        
        
        frames = [self.load_frame(frame) for frame in frames]
 
        frames = self.clip_transform(frames) # (T, 3, 224, 224)
        

        # include gaze_frame is using for gaze history and depth_frames if depth -----------------------------------------------
        gaze_frames = self.clip_transform(gaze_frames)
        depth_frames = self.clip_transform(depth_frames)
        

        #add this to instance: "depth_frames":depth_frames
        instance = {'frames':frames, 'verb':entry['verb'], 'noun':entry['noun'], 'length':length, 'gaze': padded_gaze_points, "gaze_frames":gaze_frames, "depth_frames":depth_frames}


        #--------------------------------------------------------------------------#
        # load the positive and negative images for the triplet loss
        positive, negative, pos_gaze, neg_gaze = self.select_inactive_instances(entry)
        gaze_points_pos = np.array(pos_gaze, dtype=np.float32)
        gaze_points_neg = np.array(neg_gaze, dtype=np.float32)

        
        instance.update({'positive':positive, 'negative':negative, 'pos_gaze':gaze_points_pos, 'neg_gaze':gaze_points_neg})

        #--------------------------------------------------------------------------#

        return instance

    def __len__(self):
        return len(self.data)


#----------------------------------------------------------------------------------#

import cv2
def compute_heatmap(points, image_size, k_ratio, transpose):
    """Compute the heatmap from annotated points.
    Args:
        points: The annotated points.
        image_size: The size of the image.
        k_ratio: The kernal size of Gaussian blur.
    Returns:
        The heatmap array.
    """
    points = np.asarray(points)
    heatmap = np.zeros((image_size[0], image_size[1]), dtype=np.float32)
    n_points = points.shape[0]

    for i in range(n_points):
        x = points[i, 0]
        y = points[i, 1]
        row = int(x)
        col = int(y)

        try:
            heatmap[row, col] += 1.0
        except:
            # resize pushed it out of bounds somehow
            row = min(max(row, 0), image_size[0]-1)
            col = min(max(col, 0), image_size[1]-1)
            heatmap[row, col] += 1.0
        
    # Compute kernel size of the Gaussian filter. The kernel size must be odd.
    k_size = int(np.sqrt(image_size[0] * image_size[1]) / k_ratio)
    if k_size % 2 == 0:
        k_size += 1

    # Compute the heatmap using the Gaussian filter.
    heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)

    if np.sum(heatmap)>0:
        heatmap /= np.sum(heatmap)

    if transpose:
        heatmap = heatmap.transpose()

    return heatmap    

def generate_heatmaps(annots, kernel_size, out_file, transpose):

    
    def generate(images):
        print ('Generating %d heatmaps'%len(images))
        keys = [tuple(entry['image']) + (str(entry['verb']), ) for entry in images]
        hmaps = Parallel(n_jobs=16, verbose=2)(delayed(compute_heatmap)(entry['points'], entry['shape'], kernel_size, transpose) for entry in images)
        return keys, hmaps

    train_keys, train_hmaps = generate(annots['train_images'])
    test_keys, test_hmaps = generate(annots['test_images'])

    # save the heatmaps as an h5 file
    hf = h5py.File(out_file, 'w')
    keys = [np.array(key, dtype='S') for key in train_keys+test_keys]
    hf.create_dataset('keys', data=keys, dtype=h5py.special_dtype(vlen=str))
    for idx, hmap in enumerate(train_hmaps+test_hmaps):
        hf.create_dataset('heatmaps/%d'%idx, data=hmap, dtype=np.float32)
    hf.close()


#----------------------------------------------------------------------------------#

class HeatmapLoader:
    def __init__(self, hf):
        hf = h5py.File(hf, 'r')
        self.heatmaps = hf['heatmaps']

        self.map = {tuple(k.decode() if isinstance(k, bytes) else k for k in key): str(idx) for idx, key in enumerate(np.array(hf['keys']))}

    
    def __call__(self, key):

        heatmap = self.heatmaps[self.map[key]]
        heatmap = np.array(heatmap)
        
        heat = Image.fromarray(heatmap, mode='L')  # 'L' mode for grayscale

        # Save the image as a .jpg file
        heat.save("heat_im_test.jpg")
        
    
        return heatmap

class HeatmapDataset(tdata.Dataset):

    def __init__(self, root, split, hm_file, std_norm=True):
        self.root = root
        self.split = split
        self.heatmaps = None
        self.hm_file = hm_file
        self.pair_transform = util.PairedTransform(self.split, std_norm)


    def init_hm_loader(self):
        return HeatmapLoader(self.hm_file)

    # Function to load the inactive image + its associated heatmap
    def load_image_heatmap(self, entry):
        pass

    # return the key for the entry (used for matching .h5 heatmaps)
    def key(self, entry):
        return tuple(entry['image']) + (str(entry['verb']), )
  
    def __getitem__(self, index):

        if self.heatmaps is None:
            self.heatmaps = self.init_hm_loader()

        entry = self.data[index]
        img, heatmap = self.load_image_heatmap(entry)
        instance = {'img':img, 'verb':entry['verb'], 'heatmap':heatmap}
        return instance

    def __len__(self):
        return len(self.data)
