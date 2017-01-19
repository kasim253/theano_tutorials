import numpy
import theano
import theano.tensor as T
from theano import pp
from theano import function

# Lesson 1: Simple Algebra
# create variables representing a floating-point scalar quantity with the given names
x = T.dscalar('x')
y = T.dscalar('y')
# create variable that represents addition, pretty print out
z = x + y
print('Lesson 1')
print(pp(z))
# test function
f = function([x, y], z)
print(f(2, 3))

# Lesson 2: Matrix and Vector Manipulation
# instantiate x and y using matrix types
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
print('Lesson 2')
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
x = T.dmatrix('x')
s = 1/(1+T.exp(-x))
logistic = function([x], s)
print('Lesson 3')
print(logistic([[0, 1], [-1, -2]]))

# Lesson 4: Compute more than one output
a, b = T.matrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_sq = diff**2
f = function([a, b], [diff, abs_diff, diff_sq])
print('Lesson 4')
print(f([[1, 1], [1, 1]], [[0, 1], [2, 3]]))

# Lesson 5: Default Values as Arguments
from theano import In
x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
f = function([x, In(y, value=1), In(w, value=2, name='w_by_name')], z)
print('Lesson 5')
print(f(33, 0, 1), f(33, w_by_name=1), f(33, w_by_name=1, y=0))

# Lesson 6: Using Shared Variables
from theano import shared
# create shared variable - hese are hybrid symbolic and
# non-symbolic variables whose value may be shared between multiple functions
state = shared(0)
inc = T.iscalar('inc')
# create accumulator function: at the beginning, the state is initialized to zero.
# Then, on each function call, the state is incremented by the function’s argument
# 'update' must be supplied with a list of pairs of the form (shared-variable, new expression).
accumulator = function([inc], state, updates=[(state, state+inc)], on_unused_input='ignore')
print('Lesson 6')
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
# can be useful for creating similar functions but with different shared variables or updates
state.set_value(0)
accumulator(10)
print('Lesson 7')
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
