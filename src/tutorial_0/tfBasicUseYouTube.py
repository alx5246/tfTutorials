# Alex Lonsberry
# November 2016
#
# From Dan Does Data: Tensor Flow, Basic Usage
#
# This is essentially this dude running through the "basic usage" on tensor flow documention.
#
# He did Mnist first, but says this is the tutorial you should go through first

import tensorflow as tf

# A constant in the graph will never change
matrix1 = tf.constant([[3., 3.]]) # This is an op or node in the graph
matrix2 = tf.constant([[2.],[2.]]) # This is an op or node in the graph
# Take tensors and multiply them
product = tf.matmul(matrix1,matrix2)

# Lets make an interactive session, and use .eval()
sess = tf.InteractiveSession()
print(product.eval())

# Let try the original way
result = sess.run(product)
print(result)
print(type(result))

# Make sure to close the session!!!!
sess.close()

#
sess = tf.InteractiveSession()
x = tf.Variable([1.0, 2.0])
a = tf.Variable([3.0, 3.0])
x.initializer.run() # This actually initializes the variables, they are NOT automatically initialized (Constants are however?)
a.initializer.run() # Thi
sub = tf.sub(x, a)
sub.eval()
print(sub.eval())
print(type(sub.eval()))
print(type(a))
sess.close()


# Tensors is the only thing we can pass between nodes/op of graph


# Variables, these are the things that we can hold throughout computation
print("\n")
state = tf.Variable(0., name='counter')
print(state.name)
# The 'state' variable has not been intilaized yet
# Make an Op to add one to 'state'
one = tf.constant(1.)
new_value = state + one # Trying not to use tf.add ... not sure about time penalty
update = tf.assign(state, new_value)
# Initialize
init_op = tf.initialize_all_variables() # This still does not init as we are not in a sesson
#Lauch the graph and run teh ops
sess = tf.Session()
# Run the init op
sess.run(init_op)
#
print(sess.run(state))
#print(state.eval()) # This will not work...
#
for i in range(3):
    sess.run(update)
    print(sess.run(state))








