"""198-dimensional hand feature vector adapted from the supplied notebook."""
import numpy as np
from scipy.spatial.distance import pdist

FIRST = [0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19]
SECOND = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
ANGLE_A = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18]
ANGLE_B = [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]


def extract_features(hand_landmarks):
    """21 x (xyz, visibility)=84, 15 angles, first 99 pairwise distances."""
    joint = np.zeros((21,4),dtype=np.float32)
    for j,lm in enumerate(hand_landmarks.landmark):
        joint[j] = [lm.x,lm.y,lm.z,getattr(lm,'visibility',0.0)]
    direction=joint[SECOND,:3]-joint[FIRST,:3]
    # Zero-length bone vectors are rare but should not produce NaN.
    direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-8)
    dot=np.einsum('nt,nt->n',direction[ANGLE_A],direction[ANGLE_B])
    angles=np.degrees(np.arccos(np.clip(dot,-1.0,1.0)))
    distances=pdist(joint[:,:3])[:99]
    features=np.concatenate([joint.flatten(),angles,distances]).astype(np.float32)
    assert features.shape==(198,)
    return features
