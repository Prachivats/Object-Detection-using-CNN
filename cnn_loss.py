import numpy as np

# calculating cross entropy loss
def cross_entropy_loss(inputs, class_labels):
    
    # print("loss")
    # print(inputs)
    m = len(inputs)
    #print ("class_labels",class_labels)
    #print("input " ,inputs)
    #print(" arg ",  )
    #print(inputs[range(m),class_labels.argmax(axis=1)])
    #exp_inputs = np.log(inputs[range(m),class_labels.argmax(axis=1)])
    #exp_inputs = np.log(inputs[inputs!=0])
    loss = -(1/m)*np.sum(np.multiply(class_labels,np.log(inputs)))
    #loss = -np.sum(np.log(inputs[np.arange(m), class_labels[:,1]])) / m

    return loss