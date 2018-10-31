import os
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from tnn import main
from EIcell import EICell
from SimpleRNNCell import tnn_ConvNormBasicCell
import numpy as np
from os import path
#import valset_objectABC as valss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


batch_size = 256 # batch size for training
IMAGE_SIZE = 128
NUM_CATS = 1000 # number of categories
NUM_TIMESTEPS = 0 # number of timesteps we are predicting on
NETWORK_DEPTH = 8 # number of total layers in our network
TOTAL_TIMESTEPS = NETWORK_DEPTH + NUM_TIMESTEPS # we always unroll num_timesteps after the first output of the model
TRAIN = True # the train flag, SET THIS to False if evaluating the loss
TFRec_data = '/mnt/sdb/tfrecord/'

name_str = 'alexnet_128FF'
pwr = 2.0
dt= 0 #3.0
FFdist = 'truncated_normal' #'glorot_uniform'
if FFdist == 'truncated_normal':
    std = .03
elif FFdist == 'glorot_uniform':
    std = 0 
reg = .0007
k = 0 #1.0
EIdist ='truncated_normal'
if EIdist is None:
  EIstd = 0
else:
  EIstd = .02
clip = 0 #1 to clip gradients, 0 to not
lr = .01
decay = 8e4
mom = .9
mDQ=2400
model_num = 'p'+str(pwr)+'_s'+str(NUM_TIMESTEPS)+'_'+str(dt)+'dt'+str(std)+'std_reg'+str(reg)+'_EIstd'+str(EIstd)+'_k'+str(k)+'minDQ'+str(mDQ)+'Mom'+str(mom)+'Poly'+str(decay)+'lr'+str(lr)+'clip'+str(clip)
tot_batches = 500000 #5500
save_path = '/mnt/sdb/gwl2108/'
perf_path = '/home/gwl2108/'

#wd = .0005

# we unroll at least NETWORK_DEPTH times (8 in this case) so that the input can reach the output of the network
# note tau is the value of the memory decay (by default 0) at the fc layers (due to GPU memory constraints) and trainable_flag is whether the memory decay is trainable, which by default is False


def make_test_images(n_batches):
   test_images_all = []
   test_labels_all = []
   val_ims = vals.imagenet_valsest(batch_size)
   for i in range(n_batches):     
       test_set, tlabs = val_ims.get_tset()
       test_images = test_set * (1. / 255) - 0.5
       test_labels = tlabs
       test_images_all.append(test_images)
       test_labels_all.append(test_labels)
   return test_images_all, test_labels_all 


def model_func(input_images, train=True, ntimes=TOTAL_TIMESTEPS, batch_size=batch_size, edges_arr=[], base_name='', tau=0.0, trainable_flag=False, channel_op='concat'):
    bi = 0
    with tf.variable_scope(name_str):
        base_name += '.json'
        print('Using base: ', base_name)
        # creates the feedforward network graph from json
        G = main.graph_from_json(base_name)

        for node, attr in G.nodes(data=True):
            if node in ['fc6', 'fc7']:
                if train: # we add dropout to fc6 and fc7 during training 'conv1','conv2','conv3','conv4','conv5'
                    print('Using dropout for ' + node)
                    attr['kwargs']['post_memory'][1][1]['keep_prob'] = 0.5
                else:
                    print('Not using dropout')
                    attr['kwargs']['post_memory'][1][1]['keep_prob'] = 1.0

            if False: #node in ['conv1','conv2','conv3','conv4','conv5']:
                if train: # we add dropout to fc6 and fc7 during training 
                    print('Using dropout for ' + node)
                    attr['kwargs']['pre_memory'][1][1]['keep_prob'] = 0.5
                else:
                    print('Not using dropout')
                    attr['kwargs']['pre_memory'][1][1]['keep_prob'] = 1.0

            memory_func, memory_param = attr['kwargs']['memory']
            #print(attr['kwargs']['memory'])
            if 'ei_frac' in memory_param:
                print('USING EI')
                attr['cell'] = EICell
                #print(attr['kwargs']['memory'][1]['power'], attr['kwargs']['memory'][1])
                attr['kwargs']['memory'][1]['power'] = pwr #i can set power this way
                attr['kwargs']['memory'][1]['dt'] = dt #i can set power this way
                attr['kwargs']['memory'][1]['k'] = k #i can set power this way
                #attr['kwargs']['memory'][1]['E_initializer'] = None
                #attr['kwargs']['memory'][1]['I_initializer'] = None
                attr['kwargs']['memory'][1]['E_initializer'] = EIdist
                attr['kwargs']['memory'][1]['I_initializer'] = EIdist
                attr['kwargs']['memory'][1]['E_init_kwargs'] = {'stddev':EIstd}
                attr['kwargs']['memory'][1]['I_init_kwargs'] = {'stddev':EIstd}
                #attr['kwargs']['memory'][1]['kernel_init'] = 'truncated_normal'
                #attr['kwargs']['memory'][1]['kernel_init_kwargs'] = {'stddev':.02}
                attr['kwargs']['memory'][1]['layer_norm'] = True 
                #attr['kwargs']['memory'][1]['ei_frac'] = 1.0 #i can set power this way
            elif 'shape' in memory_param:

                attr['cell'] = tnn_ConvNormBasicCell
                attr['kwargs']['memory'][1]['layer_norm'] = True 
                attr['kwargs']['memory'][1]['kernel_regularizer'] = None 
                attr['kwargs']['memory'][1]['bias_regularizer'] = None 

            #attr['kwargs']['memory'][1]['trainable'] = trainable_flag

            both = attr['kwargs']['pre_memory'][0]
            memory_param = both[1]
            if 'bias' in memory_param:
                bi+=1
                attr['kwargs']['pre_memory'][0][1]['bias'] = 0.0 #i can set bias this way
                print('setting bias ',bi)

            '''#both = attr['kwargs']['post_memory']
            #memory_param = both[1]
            #if len(both) > 0:
                #attr['kwargs']['pre_memory'][0][1]['batch_norm'] = True #i can set power this way
                #print('setting batch norm to true')
                #attr['kwargs']['post_memory'][0] = 'elu' #i can set power this way
                #print('USING ELU!!!', attr['kwargs']['post_memory'])'''

            if 'kernel_init' in memory_param:
                attr['kwargs']['pre_memory'][0][1]['kernel_init'] = FFdist 
                if FFdist == 'truncated_normal':
                    attr['kwargs']['pre_memory'][0][1]['kernel_init_kwargs'] = {'stddev':std}

            '''both = attr['kwargs']['memory'] #init the kernel for rec layers
            memory_param = both[1]
            if 'kernel_init' in memory_param:
                attr['kwargs']['memory'][1]['kernel_init'] = 'truncated_normal'
                attr['kwargs']['memory'][1]['kernel_init_kwargs'] = {'stddev':.005}'''

            #if 'kernel_init' in memory_param:
            #    attr['kwargs']['pre_memory'][0][1]['kernel_init'] = 'glorot_uniform'
            #    print('dif init')




        # add any non feedforward connections here: [('L2', 'L1')]
        G.add_edges_from(edges_arr)

        # initialize network to infer the shapes of all the parameters
        main.init_nodes(G, input_nodes=['conv1'], batch_size=batch_size, channel_op=channel_op)
        # unroll the network through time
        main.unroll(G, input_seq={'conv1': input_images}, ntimes=ntimes)

        outputs = {}
        # start from the final output of the model and num timesteps beyond that
        # for t in range(ntimes-NUM_TIMESTEPS, ntimes):
        #     idx = t - (ntimes - NUM_TIMESTEPS) # keys start at timepoint 0
        #    outputs[idx] = G.node['fc8']['outputs'][t]
        
        # alternatively, we return the final output of the model at the last timestep

        outputs[0] = G.node['fc8']['outputs'][-1]
        #c1 = G.node['fc6']['outputs'][-1]
        return outputs

def read_and_decode(filename_queue):

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image/encoded': tf.FixedLenFeature([], tf.string),
          'image/class/label': tf.FixedLenFeature([], tf.int64),
          'image/filename': tf.FixedLenFeature([], tf.string)
      })

  image = tf.image.decode_jpeg(features['image/encoded'], channels=3) 

  #mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32, shape=[1, 1, 3], name='img_mean') #[123.68, 116.779, 103.939]
  #std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32, shape=[1, 1, 3], name='img_std')
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  #print(image)
  image_shape = tf.stack([IMAGE_SIZE, IMAGE_SIZE])
  image = tf.image.resize_images(image, image_shape,method=tf.image.ResizeMethod.BILINEAR,align_corners=False)
  #image = (tf.cast(image, tf.float32)-mean)/std
  #print(image)

  # Randomly crop a [height, width] section of the image.
  #distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation. 
  # NOTE: since per_image_standardization zeros the mean and makes
  # the stddev unit, this likely has no effect see tensorflow#1458.
  distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(distorted_image)

  # Set the shapes of tensors.
  float_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
  #read_input.label.set_shape([1])
  label = tf.cast(features['image/class/label'], tf.int32)

  return float_image, label-1

def input_pipeline(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example, label = read_and_decode(filename_queue)
  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = mDQ
  capacity = min_after_dequeue + 3 * batch_size
  #example_batch, label_batch = tf.train.batch(
  #    [example, label], batch_size=batch_size)
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch, label_batch

# get random images of size 224, 224, 3 (this is where the imagenet images would be)
# typically they are 256, 256, 3, but we resize them using tf.resize_images to 224

# create the model
#x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3]) #passed in as right shape

#y_p = tf.placeholder(tf.int32, [batch_size,]) # predicting a single label for each input image

files = [TFRec_data+f for f in os.listdir(TFRec_data) if 'train' in f] 
val_files = [TFRec_data+f for f in os.listdir(TFRec_data) if 'val' in f]# if path.isfile(f)]
  #need to change this to be train! then create dif stream for val

tv_cond = tf.placeholder(tf.string)
#x_noise = tf.placeholder(tf.float32, [batch_size, 224, 224, 3]) #passed in as right shape
with tf.device('/cpu:0'):
     x_pipeline, y_pipeline = input_pipeline(files,batch_size) #MAKE FILE LIST
     x_valpipeline, y_valpipeline = input_pipeline(val_files,batch_size) #MAKE FILE LIST
def xt(): return x_pipeline
def xv(): return x_valpipeline
def yt(): return y_pipeline
def yv(): return y_valpipeline
x = tf.placeholder_with_default(tf.cond(tf.equal(tv_cond, 'train'),xt,xv),[batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
y_ = tf.placeholder_with_default(tf.cond(tf.equal(tv_cond, 'train'),yt,yv),[batch_size,])


'''handle = tf.placeholder(tf.float32, shape=[batch_size, 224, 224, 3])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# You can use feedable iterators with a variety of different kinds of iterator
# (such as one-shot and initializable iterators).
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

# The `Iterator.string_handle()` method returns a tensor that can be evaluated
# and used to feed the `handle` placeholder.
training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())'''

logits = model_func(x, train=TRAIN, ntimes=TOTAL_TIMESTEPS, batch_size=batch_size, edges_arr=[], base_name=name_str, tau=0.0, trainable_flag=False, channel_op='concat') 

'''with tf.name_scope(name_str): #was trying to weight regularize
    tf.get_variable_scope().reuse_variables() #added so i can use tensorboard (maybe dont need to do twice)
    var_names = [n.name for n in tf.get_default_graph().as_graph_def().node if "Variable" in n.op]
    all_weights = [tf.get_variable(x) for x in var_names if x[-7:]=='weights']  
    regularizers = tf.nn.l2_loss(all_weights[0])
    print(all_weights[0])
    for w in all_weights[1:]:
        print(w)
        regularizers += tf.nn.l2_loss(w)'''
with tf.name_scope(name_str):
    #tf.get_variable_scope().reuse_variables() #added so i can use tensorboard (maybe dont need to do twice)
    #var_names = [n.name for n in tf.get_default_graph().as_graph_def().node if "Variable" in n.op]
    #all_weights = [tf.get_variable(x) for x in var_names if x[-7:]=='weights']  
    all_weights = [v for v in tf.trainable_variables() if v.name[-7:]=='weights']
    regularizers = tf.nn.l2_loss(all_weights)
    for w in all_weights[1:]:
        regularizers += tf.nn.l2_loss(w)


# setup the loss (average across time, the cross entropy loss at each timepoint between model predictions and ground truth categories)
with tf.name_scope('cumulative_loss'):
    cumm_loss = tf.reduce_mean(tf.add_n([tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=y_) for logit in logits.values()]) / len(logits) + reg*regularizers) #.001

correct_prediction = tf.equal(tf.cast(tf.argmax(logits[0],1),tf.int32), y_) # I THINK THIS ONLY WORKS IF THERES ONLY ONE TIME POINT
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#decay the learning rate
global_step = tf.Variable(0, trainable=False)
#learning_rate = tf.placeholder(tf.float32) #.01 #tf.train.polynomial_decay(.001, global_step,
                                          #1e5, .0001,
                                          #power=0.5)
learning_rate = tf.train.polynomial_decay(lr, global_step,
                                          decay, .001,
                                          power=0.5)
# setup the optimizer
with tf.name_scope('optimizer'):
    #train_step = tf.train.AdamOptimizer(1e-4).minimize(cumm_loss)
    #optimiser = tf.train.GradientDescentOptimizer(1e-5)
    #train_step = optimiser.minimize(cumm_loss)

    #optimizer = tf.train.AdamOptimizer(learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=mom)
    gvs = optimizer.compute_gradients(cumm_loss)
    if clip:
       capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    else:
       capped_gvs = gvs
    train_step = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    #gradients, variables = zip(*optimizer.compute_gradients(cumm_loss))
    #gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    #optimize = optimizer.apply_gradients(zip(gradients, variables))



with tf.name_scope(name_str): #TBoard setup
    tf.get_variable_scope().reuse_variables() #added so i can use tensorboard

    #for v in variables:  #put more stuff here that i want to save
    #        if v[0:len(name_str)] == name_str and 'Adam' not in v:
    #             tf.summary.histogram(v, tf.get_variable(v)) 

    #variables = [n.name for n in tf.get_default_graph().as_graph_def().node if "Variable" in n.op]
    #check_vars = [tf.check_numerics(tf.get_variable(v),v) for v in variables]


    gradients_TB = optimizer.compute_gradients(loss=cumm_loss)
    #l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
    for gradient, variable in gradients_TB:
        tf.summary.histogram("gradients/" + variable.name, gradient)
        tf.summary.histogram("variables/" + variable.name, variable)
        #tf.check_numerics(gradient)

                 #vstr = name_str + '/' + v
            #else:
            #     vstr = v
            #tf.summary.histogram(v, tf.get_variable(vstr)) 
    #tf.summary.histogram('layer1_biases', tf.get_variable(name_str+'/L1/pre_0/bias'))
    #tf.summary.histogram('readout', tf.get_variable(name_str+'/readout/pre_1/weights'))


init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

    #print(tf.get_variable('alexnetEI_128FF/conv5/EIupdate/X')) 
tf.summary.scalar('loss', cumm_loss)
merge = tf.summary.merge_all()

#test_ims, test_labs = make_test_images(5) #RIGHT NOW THIS DOESNT SHUFFLE ANYTHING
val_batches = 5 
#lr = .01; 
plateau = 0
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
accs_losses = np.zeros((tot_batches,2))*np.nan
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter( save_path + 'EICells/'+name_str+'/'+model_num+'_tboard/', sess.graph)
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    saver = tf.train.Saver()
    print([n.name for n in tf.get_default_graph().as_graph_def().node if (("Variable" in n.op) and ("conv5" in n.name))])
    #STOP = STOP
    top_perf = 0
    #exc = tf.get_variable('alexnetEI_128FF/conv4/EIupdate/X')
    #inh  = tf.get_variable('alexnetEI_128FF/conv5/EIupdate/X')
    #print(exc)
    print(model_num)
    for i in range(tot_batches):
        #FIGURE OUT HOW TO DEAL WITH THE DATA SIZE and make this real:
        #batch_img = (np.random.randint(255,size=(batch_size, 224, 224, 3)))/255.-.5
        #noise_img = (np.random.rand(batch_size, 224, 224, 3))-.5
        #labels_batch = np.random.randint(1000, size=batch_size) # random ints between 0-999 for the labels as they are categories

        if i % 1000 == 0 or i==250:
            #print([v for v in tf.trainable_variables() if (('conv4' in v.name) or ('conv5' in v.name))])
            #E, I = sess.run([exc, inh], feed_dict={tv_cond:'train'},options=run_opts)
            #print(E[0,0,0:50],I[0,0,0:50])
            #TRAIN = False # set this to false for evaluation
            #train_loss = cumm_loss.eval(feed_dict={x: batch_img, y_: labels_batch})
            #sess.run(check_vars)
            train_loss, summary = sess.run([cumm_loss, merge], feed_dict={tv_cond:'train'},options=run_opts)
            #labz = sess.run([y_],feed_dict={x_noise: noise_img, y_p: labels_batch})

            print('step %d, training loss %g' % (i, train_loss))
            #print(lgts[0][0:10,0:5])
            #print(np.shape(logs))
            #print(np.max(logs,1))
            #print(np.min(logs))
            #print(imz[:,112,112,2])
            #np.savez('/home/millerlab/grace/EIcells/perfs/logs_'+name_str+'_'+model_num+str(i), lgts[0], labz)
            #tbatch, tlabels = mnist.get_testbatch(batch_size,1)
            acc = 0
            for t in range(val_batches):
                acc += accuracy.eval(feed_dict={tv_cond:'val'})
                #acc += accuracy.eval(feed_dict={x: imz, y_: labz})
       
            '''tbatch, tlabels = mnist.get_testbatch(batch_size,2)
            acc2 = accuracy.eval(feed_dict={x: tbatch, y_: tlabels})
            tbatch, tlabels = mnist.get_testbatch(batch_size,3)
            acc3 = accuracy.eval(feed_dict={x: tbatch, y_: tlabels})'''
            print('step ' + str(i) +', test accuracy: ' + str(acc/val_batches) )
            #train_loss = cumm_loss.eval(feed_dict={x: tbatch, y_: tlabels})
            accs_losses[i,0] = acc/val_batches  #maybe need to add 1e-8 to everything
            accs_losses[i,1] = train_loss
            #imz = labz
            #np.savez('/home/millerlab/grace/EIcells/perfs/'+name_str+'_IMzLz', test_ims,test_labs )
            np.save(perf_path + 'EICells/perfs/'+name_str+'_'+model_num, accs_losses)
            
            if acc/val_batches >= top_perf:
                top_perf = acc/val_batches
                save_path = saver.save(sess, save_path+"EICells/"+name_str+"/mod"+model_num+"_model.ckpt") #save weights
            else:
                plateau+=1
            if plateau == 10: # and lr > .005:
               #lr = lr*.75; 
               plateau = 0
            #summary = sess.run(merge) #, feed_dict={x: batch_img, y_: labels_batch}) #save analytics
            train_writer.add_summary(summary, i)

            #assert not np.any(np.isnan(labz))

            TRAIN = True # set this back to true for training
        #train_step.run(feed_dict={tv_cond:'train', learning_rate: lr}) #feed_dict={x: batch_img, y_: labels_batch})
        train_step.run(feed_dict={tv_cond:'train'})

