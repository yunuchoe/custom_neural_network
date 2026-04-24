import sys
import random
import math

def generate_xor_gaussians(n_samples, sigma=0.3):
    data = []
    labels = []
    centers = [
        ((-1.0, -1.0), 0),
        ((+1.0, +1.0), 0),
        ((-1.0, +1.0), 1),
        ((+1.0, -1.0), 1),
    ]
    samples_per_center = n_samples // 4
    for (cx, cy), label in centers:
        for _ in range(samples_per_center):
            x = random.gauss(cx, sigma)
            y = random.gauss(cy, sigma)
            data.append((x, y))
            labels.append(float(label))

    # shuffle data and labels together
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)
    return data, labels

# Base class for all neural network operations
class Node:
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, downstream_gradient):
        raise NotImplementedError()

    def zero_grad(self):
        pass

    def step(self, lr):
        pass

class Multiply(Node):
    def __init__(self, w=None):
        if w is None:
            w = random.gauss(0.0, 0.5)
        self.w = float(w)
        self.dw = 0.0
        self._x = None

    def forward(self, x):
        # Store input x for use in backward pass in self._x.
        # forward(x) = w * x.
        self._x = x # store
        output = self.w * x # compute foward value
        
        return output

    def backward(self, downstream_gradient):
        # Compute gradient with respect to input and accumulate gradient with respect to w.
        upstream_gradient = downstream_gradient*(self.w) # wrt input=x so w remains
        self.dw += ( downstream_gradient*(self._x) ) # wrt w so x remains (update dw)

        return upstream_gradient

    def zero_grad(self):
        self.dw = 0.0

    def step(self, lr):
        self.w -= lr * self.dw


class AddConstant(Node):
    def __init__(self, b=None):
        if b is None:
            b = random.gauss(0.0, 0.1)
        self.b = float(b)
        self.db = 0.0

    def forward(self, x):
        # forward(x) = x + b.
        output = x + self.b

        return output

    def backward(self, downstream_gradient):
        # Accumulate gradient with respect to b.
        upstream_gradient = downstream_gradient*1 # derivtive is 1 (input and b)
        self.db += ( downstream_gradient*1 )

        return upstream_gradient

    def zero_grad(self):
        self.db = 0.0

    def step(self, lr):
        self.b -= lr * self.db


class Add(Node):
    def forward(self, inputs):
        # forward((a, b)) = a + b.
        output = inputs[0] + inputs[1] # use tuple to add

        return output

    def backward(self, downstream_gradient):
        # upstream gradient should be a touple (grad_a, grad_b) with respect to the inputs.
        upstream_gradient = [0] * 2 # make empty list
        upstream_gradient[0] = downstream_gradient # derivtaive 1 again for both
        upstream_gradient[1] = downstream_gradient
        upstream_gradient = tuple(upstream_gradient) # make into tuple

        return upstream_gradient

    def zero_grad(self):
        pass

    def step(self, lr):
        pass


class SquaredError(Node):
    def __init__(self):
        self._pred = None
        self._target = None

    def forward(self, inputs):
        # forward((pred, target)) = (pred - target)^2.
        self._pred = inputs[0] # save pred
        self._target = inputs[1] # save target
        output = (inputs[0]-inputs[1])**2 # take diff and square

        return output

    def backward(self, downstream_gradient):
        upstream_gradient = [0] * 2 # make empty list
        upstream_gradient[0] = ( downstream_gradient * 2*(self._pred - self._target) ) # derivtaive is 2(stuff)*1
        upstream_gradient[1] = ( downstream_gradient * -2*(self._pred - self._target) ) # derivtaive is 2(stuff)*(-1)
        upstream_gradient = tuple(upstream_gradient) # make into tuple
       
        return upstream_gradient


class Sigmoid(Node):
    def __init__(self):
        self._out = None

    def forward(self, x):
        # out = 1 / (1 + exp(-x)).
        output = 1 / (1+math.exp(-x))
        self._out = output
       
        return output

    def backward(self, downstream_gradient):
        upstream_gradient = ( downstream_gradient * (self._out * (1-self._out)) ) # deriative is sigmoidx*(1-sigmoidx)
       
        return upstream_gradient

    def zero_grad(self):
        pass

    def step(self, lr):
        pass

# test cases
print("1. Multiply Node")
mul = Multiply(w=2.0)
print(f"   forward(3.0) = {mul.forward(3.0)}")
print(f"   backward(1.0) -> grad_input={mul.backward(1.0)}, dw={mul.dw}")
print()

print("2. AddConstant Node")
add_c = AddConstant(b=5.0)
print(f"   forward(3.0) = {add_c.forward(3.0)}")
print(f"   backward(1.0) -> grad_input={add_c.backward(1.0)}, db={add_c.db}")
print()

print("3. Add Node")
add = Add()
print(f"   forward((3.0, 4.0)) = {add.forward((3.0, 4.0))}")
print(f"   backward(1.0) = {add.backward(1.0)}")
print()

print("4. SquaredError Node")
se = SquaredError()
print(f"   forward((2.0, 5.0)) = {se.forward((2.0, 5.0))}")
print(f"   backward(1.0) = {se.backward(1.0)}")
print()

print("5. Sigmoid Node")
sig = Sigmoid()
print(f"   forward(0.5) = {sig.forward(0.5)}")
print(f"   backward(1.0) = {sig.backward(1.0)}")
print()


class Two_Input_Node(Node):
    def __init__(self, activation=None):
        self.w1 = Multiply()
        self.w2 = Multiply()
        self.add = Add()
        self.bias = AddConstant()
        self.activation = activation

    def forward(self, x1, x2):
        # The structure is: output = activation(w1*x1 + w2*x2 + b)

        # inputs
        input1 = self.w1.forward(x1) # multiple w1 with x1
        input2 = self.w2.forward(x2) # multiply w2 with x2

        # sum
        sum_val = self.add.forward( (input1, input2) ) # add the two inputs

        # add bias
        add_bias = self.bias.forward(sum_val) # bias calculation

        # apply activation
        if self.activation is not None:
            output = self.activation.forward(add_bias) # apply it to the acitvation
        else: # no activation
            output = add_bias # stimply take biad
        
        return output

    def backward(self, downstream_gradient):
        # Backpropagate through activation, bias, add, and multiply nodes.

        # activation
        if self.activation is not None: # activation is there
            output = self.activation.backward(downstream_gradient) # take backwards activation

            # add bias
            add_bias = self.bias.backward(output) # backwards bias

            # sum
            sum_val = self.add.backward(add_bias) # backwards add

            # inputs
            input1 = self.w1.backward(sum_val[0]) # split tuple up and tie to input
            input2 = self.w2.backward(sum_val[1])

            upstream_gradient = (input1, input2) # update
        else: # no activation
            output = downstream_gradient # simply take the gradient as is

            # add bias
            add_bias = self.bias.backward(output) # backwards bias

            # sum
            sum_val = self.add.backward(add_bias) # backwards add

            # inputs
            input1 = self.w1.backward(sum_val[0]) # split tuple up and tie to input
            input2 = self.w2.backward(sum_val[1])

            upstream_gradient = (input1, input2) # update
        
        return upstream_gradient

    def zero_grad(self):
        self.w1.zero_grad()
        self.w2.zero_grad()
        self.bias.zero_grad()
        if self.activation is not None:
            self.activation.zero_grad()

    def step(self, lr):
        self.w1.step(lr)
        self.w2.step(lr)
        self.bias.step(lr)
        if self.activation is not None:
            self.activation.step(lr)


class XORNetwork:
    def __init__(self, activation_constructor=Sigmoid):
        self.h1 = Two_Input_Node(activation=activation_constructor())
        self.h2 = Two_Input_Node(activation=activation_constructor())
        self.out = Two_Input_Node(activation=activation_constructor())

    def forward(self, x1, x2):
        # The structure is: 2 inputs -> 2 hidden neurons (h1, h2) -> 1 output neuron (out)
        h1 = self.h1.forward(x1, x2) # give h1 both x1 and x2
        h2 = self.h2.forward(x1, x2) # give h2 both x1 and x2

        output = self.out.forward(h1, h2) # compute foward output
       
        return output

    def backward(self, grad):
        backwards_grad = self.out.backward(grad) # recieve a tuple using backward of grad

        self.h1.backward(backwards_grad[0]) # update both
        self.h2.backward(backwards_grad[1])
       

    def zero_grad(self):
        self.h1.zero_grad()
        self.h2.zero_grad()
        self.out.zero_grad()

    def step(self, lr):
        self.h1.step(lr)
        self.h2.step(lr)
        self.out.step(lr)
        
# more test cases
print("1. Two Input Node")
tin = Two_Input_Node()
print(f"   forward(3.0) = {tin.forward(-1.0, 3.0)}")
print(f"   backward(1.0) -> grad_input={tin.backward(1.0)}, dw spotcheck={tin.w1.dw}")
print()

print("2. XOR Network")
xorn = XORNetwork()
print(f"   forward(3.0) = {xorn.forward(-1.0, 3.0)}")
print(f"   backward(1.0) -> grad_input={xorn.backward(1.0)}, dw spotcheck={xorn.h2.w1.dw}")
print()


# Train a neural network on XOR Gaussian classification
def train_xor(learning_rate=0.5, epochs=100, activation_constructor=Sigmoid):
    print("=" * 60)
    print("XOR Gaussian Binary Classification Demo")
    print("=" * 60)

    train_data, train_labels = generate_xor_gaussians(100, sigma=0.4)
    test_data, test_labels = generate_xor_gaussians(100, sigma=0.4)

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print()

    net = XORNetwork(activation_constructor=activation_constructor)

    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Network: 2 -> 2 (ReLU) -> 1 (Sigmoid)")
    print()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        for (x1, x2), target in zip(train_data, train_labels):
            net.zero_grad()
            pred = net.forward(x1, x2)
            loss = (pred - target) ** 2
            total_loss += loss
            if (1.0 if pred > 0.5 else 0.0) == target:
                correct += 1
            grad = 2.0 * (pred - target)
            net.backward(grad)
            net.step(learning_rate)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {total_loss/len(train_data):.4f}, Acc = {100*correct/len(train_data):.1f}%")

    print()
    print("=" * 60)
    print("Test Evaluation")
    print("=" * 60)

    test_correct = 0
    test_loss = 0.0
    for (x1, x2), target in zip(test_data, test_labels):
        pred = net.forward(x1, x2)
        test_loss += (pred - target) ** 2
        if (1.0 if pred > 0.5 else 0.0) == target:
            test_correct += 1

    print(f"Test Loss: {test_loss/len(test_data):.4f}")
    print(f"Test Accuracy: {100*test_correct/len(test_data):.1f}% ({test_correct}/{len(test_data)})")
    print()

    print("Sample predictions (first 10 test points):")
    print("-" * 50)
    for i in range(min(10, len(test_data))):
        x1, x2 = test_data[i]
        target = test_labels[i]
        pred = net.forward(x1, x2)
        pred_class = 1 if pred > 0.5 else 0
        marker = "ok" if pred_class == int(target) else "X"
        print(f"  ({x1:6.3f}, {x2:6.3f}) target={int(target)} pred={pred:.3f} class={pred_class} {marker}")
    print()


# some test results for train_xor

# default paramters: 0.5 learning rate, 100 epochs, Sigmoid activations.
train_xor() 

# increasing learning rates in powers of two
for exp in range(-1, 10): # up to 9
    train_xor(2**exp, 100, Sigmoid)

# no non-linear activations (i.e., all activations set to None).
# this sometimes causes a fail (out of range) because the random value chosen earlier results in a too large value with no activation
train_xor(0.5, 100, lambda: None)

# best tested paramters: 0.5 learning rate, 100 epochs, Sigmoid activations.
train_xor(1.5, 200, Sigmoid)
