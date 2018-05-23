import tensorflow as tf

tf.enable_eager_execution()
tf.executing_eagerly()

x1 = [[2.]]
sum1 = tf.matmul(x1, x1)

x2 = tf.constant(15,dtype=tf.int32)
x3 = tf.constant(8,dtype=tf.int32)

sum2 = x2+x3

print(sum1)
print(sum2)