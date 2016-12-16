# A. Lons
# December 2016
#
# My own version of some of the code and examples from tensorflow How-To "Sharing-Variables". Really this is like a
# sandbox where I explore the functionality!

import tensorflow as tf

########################################################################################################################
# PART 1: CANNOT MAKE TWO SETS OF VARIABLES WITH SAME NAME-SCOPE.. UNLESS REUSE
# Here I show that I cannot simply name two variables with the same name. I do this by recalling a method that inacts
# the same tf.varaible_scope for each call.

print("\nPART 1: Testing variables with same scope...")

inputVec1 = tf.constant([[1., 3., 5.]])
inputVec2 = tf.constant([[1., 3., 5.]])

def mutl_rand_mat(input, kernel_shape):

    # Create some random A Matrix
    randMat = tf.get_variable("randMat", kernel_shape, initializer=tf.random_normal_initializer())
    # Check the variable names!
    print("   The newly created matrix variable name = ", randMat.name)
    # Create some operation
    newMat = tf.matmul(input, randMat)
    return newMat

def single_filter(inputVec):

    # Multiple the randomly generated matrix by the input
    with tf.variable_scope("mat1"):
        output1 = mutl_rand_mat(inputVec, [3, 3])
    return output1

# I CANNOT run the following! Becuse I am trying to remake the same variable
#result1 = single_filter(inputVec1)
#result2 = single_filter(inputVec2)

# I CAN run the following because I do all reuse, note the "reuse" parameter is inherited, that is why this works!
with tf.variable_scope("matMults") as scope:
    result1 = single_filter(inputVec1)
    scope.reuse_variables()
    result2 = single_filter(inputVec2)

with tf.Session() as sess:

    # I have to initialize the variables (Random Matricies)
    sess.run(tf.initialize_all_variables())

    # Run Session

    print("   The outputs sould be the same, meaning the internal variables are identical!")
    result = sess.run(result1)
    print(result)
    result = sess.run(result2)
    print(result)
    print("\n####################################################################################\n")


########################################################################################################################
# PART 2: PLAYING WITH NAMES AND INHERITANCE
# Here I am replicating some of the simple examples on the How-To guid, just to see how the scopes work!. This boils
# down to creating variables and looking at their names!

print("Created a new variable v with the variable scope 'foo', what is v.name?")
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
print("   v.name = ", v.name)

print("\nCreated two new variables with variable scope 'oof', do they share the same name?")
with tf.variable_scope("oof"):
    v1 = tf.get_variable("v", [1])
with tf.variable_scope("oof", reuse=True):
    v2 = tf.get_variable("v", [1])
print("   v1.name = ", v1.name)
print("   v2.name = ", v2.name)

print("\nMixing variable_scope, 'Bpp', and name_scope, 'Nre', how does the variable v3 get named? How about the op x?")
print("In this case we name the variable scope first, and then the name scope")
with tf.variable_scope("Bpp"):
    with tf.name_scope("Nre"):
        v3 = tf.get_variable("v", [1])
        x = tf.add(1.0, v3)
        print("   v3.name = ", v3.name)
        print("   x.name = ", x.name)

print("\nMixing variable_scope, 'Bpp', and name_scope, 'Nre', how does the variable v3 get named? How about the op x?")
print("In this case we name the name scope first, and then the variable scope")
with tf.name_scope("NRe"):
    with tf.variable_scope("BPp"):
        v3 = tf.get_variable("v", [1])
        x = tf.add(1.0, v3)
        print("   v3.name = ", v3.name)
        print("   x.name = ", x.name)