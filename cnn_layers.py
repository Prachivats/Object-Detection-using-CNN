import numpy as np

# Creating the convolutional layer Class


class Convolution_Layer:

    def __init__(self, inputs_channel, num_filters, kernel_size, padding, stride, learning_rate, name):
        # weight size: (F, C, K, K)
        # bias size: (F)
        print ("I am in init of conv")
        self.Filters = num_filters
        self.Kernels = kernel_size
        self.Channels = inputs_channel

        self.weights = np.random.randn(
            self.Filters, self.Kernels, self.Kernels, self.Channels)
        self.bias = np.random.randn(self.Filters, 1)
        # # Weights initialization
        # for i in range(0,self.Filters):
        #     self.weights[i,:,:,:] = np.random.normal(loc=0, scale=np.sqrt(1./(self.Channels*self.Kernels*self.Kernels)), size=(self.Channels, self.Kernels, self.Kernels))

        self.pad = padding
        self.ST = stride
        self.LRate = learning_rate
        self.Layer_name = name

    # Forward Propagation function
    def forward_propagation(self, inputs):
        """
        this forward is doing the convolution and storing the results into the feature maps
        (Calculating the feature maps)
        """
        #print("conv layer",inputs)
        self.inputs = inputs
        (m, n_H_prev, n_W_prev, n_C_prev) = inputs.shape
        # Zero padding
        # print (inputs.shape)
        pad_input = np.pad(inputs, ((0, 0), (self.pad, self.pad),
                                    (self.pad, self.pad), (0, 0)), 'constant', constant_values=0)
        # print (pad_input.shape)
        n_H = int((n_H_prev - self.Kernels + 2 * self.pad) / self.ST) + 1
        n_W = int((n_W_prev - self.Kernels + 2 * self.pad) / self.ST) + 1
        # input size: (C, W, H)
        # output size: (N, F ,WW, HH)
        # Channels = inputs.shape[3]
        # Width = inputs.shape[1]+2*self.pad
        # Height = inputs.shape[2]+2*self.pad
        # self.inputs = np.zeros(( Width, Height, Channels))
        # for c in range(inputs.shape[2]):
        #     self.inputs[:,:, c] = self.zero_pad(inputs[:,:, c], self.pad)
        # WW = int((Width - self.Kernels)/self.ST) + 1
        # HH = int((Height - self.Kernels)/self.ST) + 1
        # print("width and height are:", WW, HH)
        conv_output = np.zeros((m, n_H, n_W, self.Filters))
        # print("the shape of the inputs is",self.inputs.shape)
        # print("the shape of the weights is" , self.weights.shape)

        # Doing the Convolution
        try:
            for i in range(m):
                pad_input_single = pad_input[i]
                # print (pad_input_single.shape)
                for h in range(n_H):
                    for w in range(n_W):
                        for c in range(self.Filters):
                            vert_start = h * self.ST
                            vert_end = vert_start + self.Kernels
                            horiz_start = w * self.ST
                            horiz_end = horiz_start + self.Kernels
                            a_slice_prev = pad_input_single[vert_start:vert_end,
                                                            horiz_start:horiz_end, :]
                            # print (a_slice_prev.shape)
                            # print(self.weights[c,:, :, :].shape)
                            # print (np.multiply(a_slice_prev, self.weights[c,:, :, :]).shape)
                            conv_output[i, h, w, c] = np.sum(np.multiply(
                                a_slice_prev, self.weights[c, :, :, :])) + self.bias[c]

                        # print(feature_maps[w,h,f])
        except IndexError:
            print("I m having Index error")
        # print("the feauture map shape is:",feature_maps.shape )
        #print ("conv_output",conv_output)
        return conv_output

    # Forward Propagation function.
    def backward_propagation(self, dy):

        #Width, Height, Channels = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)

        m, n_W, n_H, F = dy.shape
        # Pad A_prev and dA_prev
        A_prev_pad = np.pad(self.inputs, ((0, 0), (self.pad, self.pad),
                                          (self.pad, self.pad), (0, 0)), 'constant', constant_values=0)
        dA_prev_pad = np.pad(dx, ((0, 0), (self.pad, self.pad),
                                  (self.pad, self.pad), (0, 0)), 'constant', constant_values=0)

        for i in range(m):
                pad_input_single = A_prev_pad[i]
                da_prev_pad = dA_prev_pad[i]
                # print (pad_input_single.shape)
                for h in range(n_H):
                    for w in range(n_W):
                        for c in range(F):
                            vert_start = h
                            vert_end = vert_start + self.Kernels
                            horiz_start = w
                            horiz_end = horiz_start + self.Kernels
                            a_slice_prev = pad_input_single[vert_start:vert_end,
                                                            horiz_start:horiz_end, :]
                            da_prev_pad[vert_start:vert_end, horiz_start:horiz_end,
                                        :] += self.weights[c, :, :, :] * dy[i, h, w, c]
                            dw[c,:,:,:] += a_slice_prev * dy[i, h, w, c]
                            db[c] += dy[i, h, w, c]

                dx[i, :, :, :] = da_prev_pad[self.pad:-
                                             self.pad, self.pad:-self.pad, :]

        self.weights -= self.LRate * dw
        self.bias -= self.LRate * db
#         print("the weights after backptop: ", self.weights)
        return dx
        

        # print("the shapes for dw, dx, dy are: ",dw.shape, dx.shape, dy.shape)
        

#     # Function for extracting weights and biases
#     def extract_weights(self):
#         return {self.Layer_name +'.weights':self.weights, self.Layer_name +'.bias':self.bias}

#     # Function for feeding or filling weights and biases
#     def feed_weights(self, weights, bias):
#         self.weights = weights
#         self.bias = bias

# Creating the ReLu Activation Class


class ReLu_Activation:
    def __init__(self):
        pass

    # Forward Propagation function
    def forward_propagation(self, inputs):
        print("relu")
        #print(inputs)
        self.inputs = inputs
        relu = inputs.copy()
        relu[relu < 0] = 0
        return relu

    # Backward Propagation function
    def backward_propagation(self, da):
        dx = da.copy()
        dx[self.inputs < 0] = 0
        return dx

    # # Function for extracting weights and biases
    # def extract_weights(self):
    #     return


class Maxpool_Layer:

    def __init__(self, pool_size, stride, name):
        self.pool = pool_size
        self.ST = stride
        self.Layer_name = name
    # Forward Propagation function

    def forward_propagation(self, inputs):
        print("max layer")
        #print(inputs)
        try:
            # print("In the forward pass of maxpooling")
            (m, n_H_prev, n_W_prev, n_C_prev) = inputs.shape
            n_C = n_C_prev
            # print("the object types are", type(Channels),type(W),type(H))
            n_H = int((n_H_prev - self.pool) / self.ST) + 1
            n_W = int((n_W_prev - self.pool) / self.ST) + 1
            self.inputs = inputs
            # print("the range are ",l1,l2, C)
            max_pool = np.zeros((m,  n_H, n_W, n_C))
            # print("In maxpooling, the value of C,W,H,new_width,new_height", C,W,H,new_width,new_height)
            for i in range(m):
                for h in range(n_H):
                    for w in range(n_W):
                        for c in range(n_C):
                            vert_start = h * self.ST
                            vert_end = vert_start + self.pool
                            horiz_start = w * self.ST
                            horiz_end = horiz_start + self.pool
                            a_slice_prev = inputs[i, vert_start:vert_end,
                                                  horiz_start:horiz_end, c]
                            # print (a_slice_prev.shape)
                            # print(self.weights[c,:, :, :].shape)
                            # print (np.multiply(a_slice_prev, self.weights[c,:, :, :]).shape)
                            max_pool[i, h, w, c] = np.max(a_slice_prev)

                        # print(feature_maps[w,h,f])
        except IndexError:
            print("I m having Index error")
        # print("the feauture map shape is:",feature_maps.shape )
        return max_pool

    # Backward Propagation function
    def backward_propagation(self, dy):
        #print("maxpool backprop input",dy)
        # print (np.max(dy))
        (m, n_H, n_W, n_C) = dy.shape
        #(m, n_H_prev, n_W_prev, n_C_prev) = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        for i in range(m):
            a_prev = self.inputs[i]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = h
                        vert_end = vert_start + self.pool
                        horiz_start = w
                        horiz_end = horiz_start + self.pool
                        a_prev_slice = a_prev[vert_start:vert_end,
                                              horiz_start:horiz_end, c]
                        mask = a_prev_slice == np.max(a_prev_slice)
                        # print ("mask" , mask)
                        dx[i, vert_start:vert_end, horiz_start:horiz_end,
                            c] += np.multiply(mask, dy[i, h, w, c])
        #print("maxpool backprop output",dx)
        return dx

    # # Function for extracting weights and biases
    # def extract_weights(self):
    #     return


class Flattening_Layer:
    def __init__(self):
        pass

    # Forward Propagation function
    def forward_propagation(self, inputs):
        print("flat layer")
        #print(inputs)
        self.m, self.Width, self.Height, self.Channels, = inputs.shape
        # print (inputs[0,:,:,:])
        # print (inputs.reshape(1, self.Channels*self.Width*self.Height))
        return inputs.reshape(self.m, self.Channels*self.Width*self.Height)

    # Backward Propagation function
    def backward_propagation(self, dy):
        return dy.reshape(self.m, self.Width, self.Height, self.Channels)

    # # Function for extracting weights and biases
    # def extract_weights(self):
    #     return


class FullyConnected_Layer:

    def __init__(self, num_inputs, num_outputs, learning_rate, name):
        self.weights = 0.01*np.random.randn(num_outputs, num_inputs)
        self.bias = np.zeros((num_outputs, 1))
        self.LRate = learning_rate
        self.Layer_name = name

    # Forward Propagation function
    def forward_propagation(self, inputs):
        print("FC layer")
        #print(inputs)
        self.inputs = inputs
        return np.dot(self.inputs, self.weights.T) + self.bias.T

    # Backward Propagation function
    def backward_propagation(self, dz):
        self.m = len(self.inputs)
        dw = np.dot(dz.T, self.inputs)/self.m
        db = np.sum(dz, axis=0, keepdims=True)/self.m
        dx = np.dot(dz,self.weights)
        self.weights -= self.LRate * dw
        #print("####",self.bias.shape," db ",db.shape)
        self.bias -= self.LRate * db.T
        return dx

    # # Function for extracting weights and biases
    # def extract_weights(self):
    #     return {self.Layer_name +'.weights':self.weights, self.Layer_name +'.bias':self.bias}

    # # Function for feeding or filling weights and biases
    # def feed_weights(self, weights, bias):


class Softmax_Layer:
    def __init__(self):
        pass
    # Forward Propagation function

    def forward_propagation(self, inputs):
        #inputs[inputs > 709] = 709 
        # print("softmax")
        print(inputs)
        # exp_inputs = np.exp(inputs - np.max(inputs))
        # exp_inputs[exp_inputs == 0] = 0.0002  # shift values
        # print("exp_inputs")
        # print(exp_inputs)
        # #exp_inputs = np.exp(inputs)
        # #exp_inputs[~np.isfinite(exp_inputs)] = 0
        # sum_inputs = np.sum(exp_inputs, axis=0, keepdims=True)
        # self.out = exp_inputs/sum_inputs
        # print(" sum_inputs",sum_inputs)
        # print("self.out")
        # print(self.out)
        # return self.out
        self.m = len(inputs)

        x = (inputs - np.max(inputs, axis=1, keepdims=True))
        #x[x < -709] = -709
        exp_inputs = np.exp(x)
        #print("softmax inputs", inputs)
        #print ("exp_inputs",exp_inputs)
        sum_inputs = np.sum(exp_inputs, axis=1, keepdims=True)
        #print (sum_inputs)
        self.out = exp_inputs/sum_inputs
        #print("softmax self.out", self.out)
        #self.out[self.out == 0] = 0.000002
        return self.out

    # Backward Propagation function
    def backward_propagation(self, y):
        return self.out - y

    # # Function for extracting weights and biases
    # def extract_weights(self):
    #     return
