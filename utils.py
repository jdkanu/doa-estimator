import os
import numpy as np
import pysound as ps
import ir
from batch_convolver import convolve
from feature_extractor import extract_feature


def save_rectangle_room(dims, filepath_out='rect.obj'):
    with open(filepath_out, 'w') as ofile:
        ofile.write("o Rect\n")
        for w in [0, dims[0]]:
            for l in [0, dims[1]]:
                for h in [0, dims[2]]:
                    line = "v {:.4f} {:.4f} {:.4f}\n".format(w, l, h)
                    ofile.write(line)
        ofile.write("vn -1.0000 0.0000 0.0000\n"
                    "vn 0.0000 1.0000 0.0000\n"
                    "vn 1.0000 0.0000 0.0000\n"
                    "vn 0.0000 -1.0000 0.0000\n"
                    "vn 0.0000 0.0000 -1.0000\n"
                    "vn 0.0000 0.0000 1.0000\n"
                    "f 2//1 3//1 1//1\n"
                    "f 4//2 7//2 3//2\n"
                    "f 8//3 5//3 7//3\n"
                    "f 6//4 1//4 5//4\n"
                    "f 7//5 1//5 3//5\n"
                    "f 4//6 6//6 8//6\n"
                    "f 2//1 4//1 3//1\n"
                    "f 4//2 8//2 7//2\n"
                    "f 8//3 6//3 5//3\n"
                    "f 6//4 2//4 1//4\n"
                    "f 7//5 5//5 1//5\n"
                    "f 4//6 2//6 6//6")
    ofile.close()


def save_feature(meshpath, speechpath, src_coord, lis_coord, absorb=0.1):
    try:
        mesh = ps.loadobj(meshpath, os.path.join(os.path.dirname(meshpath), ''), absorb)
        scene = ps.Scene()

        scene.setMesh(mesh)
        src = ps.Source(src_coord)
        src.radius = 0.01
        lis = ps.Listener(lis_coord)
        lis.radius = 0.01
        lis.channel_layout_type = ps.ChannelLayoutType.ambisonic

        res = ir.MultiSoundBuffer(**scene.computeMultichannelIR(src, lis))
        wavname = meshpath.replace('.obj', '.wav')
        if res.get_length() > 0.05:
            res.save(wavname)
        else:
            print("invalid IR, abort!")
            return None
        convolved_path = meshpath.replace('.obj', '_conv.wav')
        convolve(wavname, speechpath, convolved_path)
        feature_path = convolved_path.replace('.wav', '.npy')
        extract_feature(convolved_path, feature_path)
        return feature_path
    except Exception as e:
        print(str(e))
        return None


if __name__ == "__main__":
    # save_rectangle_room([5.3,5.9,2.38])
    feature_path = save_feature('data/room.obj', 'data/84-121123-0000.flac', [1,1,1], [3,5,4])
    if feature_path:
        features = np.load(feature_path)