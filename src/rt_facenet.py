# -*- coding: utf-8 -*-

"""Performs face alignment and face verify
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
from scipy import misc
import facenet
import align.detect_face
import time

def main(args):
    """ 主函数 """
    cam = cv2.VideoCapture(0)
    frame_width = cam.get(3)
    #frame_height = cam.get(4)

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

            minsize = int(frame_width / 10) # minimum size of face
            threshold = [0.6, 0.7, 0.7] # three step's threshold
            factor = 0.709 # scale factor

            facenet.load_model('~/models/facenet/20170216-091149')
            images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
            embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')

            master_emb = None
            while True:
                grabbed, frame = cam.read()
                if grabbed:
                    if frame.ndim >= 2:
                        if frame.ndim == 2:
                            frame = facenet.to_rgb(frame)
                        frame = frame[:, :, 0:3]

                        bounding_boxes, _ = align.detect_face.detect_face(frame,
                                                                          minsize,
                                                                          pnet, rnet, onet,
                                                                          threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces > 0:
                            frame_size = np.asarray(frame.shape)[0:2]
                            face = np.zeros(4, dtype=np.int32)
                            for bbox in bounding_boxes:
                                face[0] = np.maximum(bbox[0], 0)
                                face[1] = np.maximum(bbox[1], 0)
                                face[2] = np.minimum(bbox[2], frame_size[1])
                                face[3] = np.minimum(bbox[3], frame_size[0])

                                cropped = frame[face[1]:face[3], face[0]:face[2], :]
                                scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
                                left_top_corner = (face[0], face[1])
                                right_bottom_corner = (face[2], face[3])
                                cv2.rectangle(frame, left_top_corner, right_bottom_corner,
                                              (0, 255, 0), 2)
                                feed_dict = {images_placeholder:[scaled],
                                             phase_train_placeholder:False}
                                emb = sess.run(embeddings, feed_dict=feed_dict)[0]

                                if master_emb is None:
                                    master_emb = emb
                                else:
                                    dist = cosine(emb, master_emb)
                                    edist = np.sum(np.square(np.subtract(emb, master_emb)))
                                    cv2.putText(frame, '%.2f-%.2f-%.2f' % (bbox[4], dist, edist),
                                                (face[0] + 10, face[1] + 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                        cv2.imshow('Realtime Face Recognizer', frame)

                if cv2.waitKey(15) & 0xFF == ord('q'):
                    break

            cam.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main(None)
