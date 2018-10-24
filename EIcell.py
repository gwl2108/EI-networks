from tnn.cell import *
import numpy as np

# super class for convolutional RNNs
class ConvRNNCell(object):
  """Abstract object representing an Convolutional RNN cell.
  """

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.
    """
    raise NotImplementedError("Abstract method")

  @property
  def state_size(self):
    """size(s) of state(s) used by this cell.
    """
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      tensor of shape '[batch_size x shape[0] x shape[1] x out_depth]
      filled with zeros
    """
    shape = self.shape
    out_depth = self._out_depth
    zeros = tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype) 
    return zeros

# according to the grant, the power is typically between 2 - 5, so I by default set it to 2
class EIupdate(ConvRNNCell):
  def __init__(self,
               shape,
               filter_size, 
               out_depth,
               power=2.0,
               ei_frac=0.5,
               dt=0.1,
               k=.04,
               E_initializer='truncated_normal',
               I_initializer='truncated_normal',
               E_init_kwargs= {'stddev':.02},
               I_init_kwargs= {'stddev':.02},
               activation=tf.nn.relu,
               bias_initializer=None,
               layer_norm=False):

    """Initialize the EI cell.
    Args:
      shape: int tuple thats the height and width of the cell
      filter_size: int tuple thats the height and width of the filter
      out_depth: int thats the depth of the cell 
      activation: Activation function of the inner states, set to be a power law.
    """
    print(dt, E_initializer,I_initializer)

    self.shape = shape
    self.filter_size = filter_size
    self._out_depth = out_depth 
    self._size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
    self._activation = activation or tf.nn.relu
    self._E_initializer = E_initializer
    self._I_initializer = I_initializer
    self._E_init_kwargs = E_init_kwargs
    self._I_init_kwargs = I_init_kwargs
    self._bias_initializer = bias_initializer
    self._power = power
    self._ei_frac = ei_frac
    self._k = k
    self._dt = dt
    self._layer_norm = layer_norm

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size 

  def __call__(self, inputs, state):
    with tf.variable_scope(type(self).__name__): # EIupdate
      W_out = _conv_linear([state], self.filter_size, self._out_depth, False, E_initializer=self._E_initializer, I_initializer=self._I_initializer, E_init_kwargs=self._E_init_kwargs, I_init_kwargs=self._I_init_kwargs, ei_frac=self._ei_frac)

      #T_inv = tf.get_variable(name="T_inv", shape=[self.shape[0], self.shape[1], self._out_depth], dtype=W_out.dtype)
      T_inv = np.ones(self._out_depth)/20.; T_inv[int(self._out_depth*self._ei_frac):] = (1/10.) #taus for E and I
      #T_inv = tf.get_variable(name="T_inv", shape=[self.shape[0], self.shape[1], self._out_depth], dtype=W_out.dtype,trainable=False)

      inner_term = self._k*tf.pow(self._activation(W_out + inputs), self._power) - state
      outer_const = self._dt*T_inv # an alternative to the timeconstant instead of inverting

      if self._layer_norm:
         new_state = tf.contrib.layers.layer_norm(outer_const*inner_term + state, reuse=tf.AUTO_REUSE, scope='layer_norm')
                                    
      else:
          new_state = outer_const*inner_term + state
      return new_state, new_state # the output and the state are the same

class EICell(ConvRNNCell):

    def __init__(self,
                 harbor_shape,
                 harbor=(harbor, None),
                 pre_memory=None,
                 memory=(memory, None),
                 post_memory=None,
                 input_init=(tf.zeros, None),
                 state_init=(tf.zeros, None),
                 dtype=tf.float32,
                 name=None
                 ):

        self.harbor_shape = harbor_shape
        self.harbor = harbor if harbor[1] is not None else (harbor[0], {})
        self.pre_memory = pre_memory
        self.memory = memory if memory[1] is not None else (memory[0], {})
        self.post_memory = post_memory

        self.input_init = input_init if input_init[1] is not None else (input_init[0], {})
        self.state_init = state_init if state_init[1] is not None else (state_init[0], {})

        self.dtype = dtype
        self.name = name

        self._reuse = None
       
        self.internal_cell =  EIupdate(memory[1]['shape'], memory[1]['filter_size'], memory[1]['out_depth'], memory[1]['power'], memory[1]['ei_frac'], memory[1]['dt'],memory[1]['k'],memory[1]['E_initializer'],memory[1]['I_initializer'],memory[1]['E_init_kwargs'],memory[1]['I_init_kwargs'])

    def __call__(self, inputs=None, state=None):
        """
        Produce outputs given inputs
        If inputs or state are None, they are initialized from scratch.
        :Kwargs:
            - inputs (list)
                A list of inputs. Inputs are combined using the harbor function
            - state
        :Returns:
            (output, state)
        """
        # if hasattr(self, 'output') and inputs is None:
        #     raise ValueError('must provide inputs')

        # if inputs is None:
        #     inputs = [None] * len(self.input_shapes)
        # import pdb; pdb.set_trace()

        with tf.variable_scope(self.name, reuse=self._reuse):
            # inputs_full = []
            # for inp, shape, dtype in zip(inputs, self.input_shapes, self.input_dtypes):
            #     if inp is None:
            #         inp = self.output_init[0](shape=shape, dtype=dtype, **self.output_init[1])
            #     inputs_full.append(inp)

            if inputs is None:
                inputs = [self.input_init[0](shape=self.harbor_shape,
                                             **self.input_init[1])]
            output = self.harbor[0](inputs, self.harbor_shape, self.name, reuse=self._reuse, **self.harbor[1])

            pre_name_counter = 0
            for function, kwargs in self.pre_memory:
                with tf.variable_scope("pre_" + str(pre_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output = function(output, inputs, **kwargs) # component_conv needs to know the inputs
                    else:
                       output = function(output, **kwargs)
                pre_name_counter += 1
                
            if state is None:
                state = self.internal_cell.zero_state(output.get_shape().as_list()[0], dtype = self.dtype)
            
            output, state = self.internal_cell(output, state)
            self.state = tf.identity(state, name='state')

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output = function(output, inputs, **kwargs)
                    else:
                       output = function(output, **kwargs)
                post_name_counter += 1
            self.output = tf.identity(tf.cast(output, self.dtype), name='output')
            # scope.reuse_variables()
            self._reuse = True
        self.state_shape = self.state.shape
        self.output_shape = self.output.shape
        return self.output, self.state

    @property
    def state_size(self):
        """
        Size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        # if self.state is not None:
        return self.state_shape
        # else:
        #     raise ValueError('State not initialized yet')

    @property
    def output_size(self):
        """
        Integer or TensorShape: size of outputs produced by this cell.
        """
        # if self.output is not None:
        return self.output_shape
        # else:
        #     raise ValueError('Output not initialized yet')

def _conv_linear(args, filter_size, out_depth, bias, bias_initializer=None, E_initializer=None, I_initializer=None, E_init_kwargs=None,  I_init_kwargs=None, ei_frac=0.5):
  """convolution:
  Args:
    args: a 4D Tensor or a list of 4D, batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    out_depth: int, number of features.
    bias: boolean as to whether to have a bias.
    bias_initializer: starting value to initialize the bias.
    E_initializer: starting value to initialize the excitatory weights.
    I_initializer: starting value to initialize the inhibitory weights.
    ei_frac: fraction of input channels dedicated to be excitatory cells (default is 0.5)
  Returns:
    A 4D Tensor with shape [batch h w out_depth]
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 4:
      raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
    if not shape[3]:
      raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
    else:
      total_arg_size_depth += shape[3]

  dtype = [a.dtype for a in args][0]

  # Now the computation. 
  #print(I_initializer, type(I_initializer), I_initializer=='truncated_normal')
  num_E = (int)(np.floor(ei_frac*total_arg_size_depth))
  num_I = total_arg_size_depth - num_E
  if E_initializer is None:
      X = tf.get_variable("X", [filter_size[0], filter_size[1], num_E, out_depth], dtype=dtype, initializer=E_initializer)
      print('not using Enorm')
  elif E_initializer=='truncated_normal':
      X = tf.get_variable("X", [filter_size[0], filter_size[1], num_E, out_depth], dtype=dtype, initializer=tf.truncated_normal_initializer(**E_init_kwargs))
      print('using Enorm')
  if I_initializer is None:
      Y = tf.get_variable("Y", [filter_size[0], filter_size[1], num_I, out_depth], dtype=dtype, initializer=I_initializer)
      print('not using Inorm')
  elif I_initializer=='truncated_normal':
      Y = tf.get_variable("Y", [filter_size[0], filter_size[1], num_I, out_depth], dtype=dtype, initializer=tf.truncated_normal_initializer(**I_init_kwargs))
      print('using Inorm')
  W_E = tf.nn.relu(X) # excitatory weights (> 0)
  W_I = -1.0*tf.nn.relu(Y) # inhibitory weights (< 0)
  kernel = tf.concat([W_E, W_I], axis=2) # concatenate along input channel dimension
  if len(args) == 1:
    res = tf.nn.conv2d(args[0], kernel, strides=[1, 1, 1, 1], padding='SAME')
  else:
    res = tf.nn.conv2d(tf.concat(axis=3, values=args), kernel, strides=[1, 1, 1, 1], padding='SAME')
  if not bias:
    return res
  if bias_initializer is None:
    bias_initializer = tf.constant_initializer(0.0, dtype=dtype)

  bias_term = tf.get_variable(
      "bias", [out_depth],
      dtype=dtype,
      initializer=bias_initializer)
  return res + bias_term

