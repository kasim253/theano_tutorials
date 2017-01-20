from __future__ import print_function
import numpy
import theano
import theano.tensor as T
from theano import pp
from theano import function

# Lesson 1: Simple Algebra
print('Lesson 1')
# create variables representing a floating-point scalar quantity with the given names
x = T.dscalar('x')
y = T.dscalar('y')
# create variable that represents addition, pretty print out
z = x + y
print(pp(z))
# test function
f = function([x, y], z)
print(f(2, 3))

# Lesson 2: Matrix and Vector Manipulation
print('Lesson 2')
# instantiate x and y using matrix types
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
print(f([[1, 2], [3, 4]], [[10, 20], [30, 40]]))

# declare variable
a = T.vector('a')
b = T.vector('b')
# build symbolic expression
out = a ** 2 + b ** 2 + 2 * a * b
# compile function
f = function([a, b], out)
print(f([1, 2], [4, 5]))

# Lesson 3: Logistic Function
print('Lesson 3')
x = T.dmatrix('x')
s = 1/(1+T.exp(-x))
logistic = function([x], s)
print(logistic([[0, 1], [-1, -2]]))

# Lesson 4: Compute more than one output
print('Lesson 4')
a, b = T.matrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_sq = diff**2
f = function([a, b], [diff, abs_diff, diff_sq])
print(f([[1, 1], [1, 1]], [[0, 1], [2, 3]]))

# Lesson 5: Default Values as Arguments
print('Lesson 5')
from theano import In
x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
f = function([x, In(y, value=1), In(w, value=2, name='w_by_name')], z)
print(f(33, 0, 1), f(33, w_by_name=1), f(33, w_by_name=1, y=0))

# Lesson 6: Using Shared Variables
print('Lesson 6')
from theano import shared
# create shared variable - hese are hybrid symbolic and
# non-symbolic variables whose value may be shared between multiple functions
state = shared(0)
inc = T.iscalar('inc')
# create accumulator function: at the beginning, the state is initialized to zero.
# Then, on each function call, the state is incremented by the function’s argument
# 'update' must be supplied with a list of pairs of the form (shared-variable, new expression).
accumulator = function([inc], state, updates=[(state, state+inc)], on_unused_input='ignore')
print(state.get_value())
accumulator(3)
print(state.get_value())
state.set_value(-1)
accumulator(10)
print(state.get_value())

decrementor = function([inc], state, updates=[(state, state-inc)])
decrementor(2)
print(state.get_value())

# use shared variable, but not its value using 'givens'
# 'givens' is as a mechanism that allows you to replace any part of your formula
# with a different expression that evaluates to a tensor of same shape and dtype.
state.set_value(0)
fn_of_state = state * 2 + inc
# type of 'foo' must match type of state
foo = T.scalar(dtype=state.dtype)
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
print(skip_shared(1, 3))
print(state.get_value())

# Lesson 7: Copying Functions
print('Lesson 7')
# can be useful for creating similar functions but with different shared variables or updates
state.set_value(0)
accumulator(10)
print(state.get_value())
new_state = shared(0)
new_accumulator = accumulator.copy(swap={state: new_state})
new_accumulator(100)
print(new_state.get_value())
print(state.get_value())
# remove updates
null_accumulator = accumulator.copy(delete_updates=True)
null_accumulator(900)
print(state.get_value())

# Lesson 8: Random Numbers
print('Lesson 8')
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=234)              # RandomStreams object - a random number generator
# create random stream of 2x2 matrices of draws from a uniform distribution
rv_u = srng.uniform((2,2))
# create random stream of 2x2 matrices of draws from a normal distribution
rv_n = srng.normal((2,2))
f = function([], rv_u)
print(f())
print(f())
# 'no_default_update=True' - the random number generator state is not affected by calling the returned function.
# So, for example, calling g multiple times will return the same numbers.
g = function([], rv_n, no_default_updates=True)         # Not updating rv_n.rng
print(g())
print(g())
# a random variable is drawn at most once during any single function execution:
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)     # should return 0

# seeding streams - random variables can be seeded individually or collectively
rng_val = rv_u.rng.get_value(borrow=True)   # Get the rng for rv_u
rng_val.seed(89234)                         # seeds the generator
rv_u.rng.set_value(rng_val, borrow=True)    # Assign back seeded rng

srng.seed(902340)  # seeds rv_u and rv_n with different seeds each

# sharing streams between functions
state_after_v0 = rv_u.rng.get_value().get_state()
print(nearly_zeros())     # this affects rv_u's generator
v1 = f()
print(v1)
rng = rv_u.rng.get_value(borrow=True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng, borrow=True)
v2 = f()             # v2 != v1
print(v2)
v3 = f()             # v3 == v1

# copying random state between theano graphs
# might arise for example if you are trying to initialize the state of a model,
# from the parameters of a pickled version of a previous model
# Each time a random variable is drawn from a RandomStreams object, a tuple is added to the state_updates list.
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams


class Graph():
    def __init__(self, seed=123):
        self.rng = RandomStreams(seed)
        self.y = self.rng.uniform(size=(1,))

g1 = Graph(seed=123)
f1 = function([], g1.y)
g2 = Graph(seed=987)
f2 = function([], g2.y)


def copy_random_state(g1, g2):
    if isinstance(g1.rng, MRG_RandomStreams):
        g2.rng.rstate = g1.rng.rstate
    for (su1, su2) in zip(g1.rng.state_updates, g2.rng.state_updates):
        su2[0].set_value(su1[0].get_value())

# copy the state of the theano random number generators:
copy_random_state(g1, g2)
print(f1())
print(f2())