# NOTES: the concepts behind Project 2

> As in Project 1, these notes are a study guide. The real exercise is to
> **redo each derivation on paper without looking**, especially section 2,
> which the project spec asks you to derive yourself. Then compare.

Notation: N = batch size, D = features, K = classes, L = scalar mean loss over
the batch. For any array A, **Ā ≡ ∂L/∂A** has the same shape as A.

---

## 1. Vectorized backprop for a Linear layer

Forward: **Y = X W + b**, with X (N, D_in), W (D_in, D_out), b (D_out,), Y (N, D_out).

Elementwise, Y[n, j] = Σᵢ X[n, i] W[i, j] + b[j]. Apply Project 1's rule
(sum over every use) to each parameter:

- W[i, j] is used in Y[n, j] for **every** sample n, with local derivative
  X[n, i]. So W̄[i, j] = Σₙ X[n, i] Ȳ[n, j], i.e. **W̄ = Xᵀ Ȳ**, with shape
  (D_in, N)(N, D_out) = (D_in, D_out). ✓
- b[j] is broadcast to all N rows, so **b̄ = Σₙ Ȳ[n, :]** (a sum, not a mean).
  The 1/N of the mean loss is already inside Ȳ.
- X[n, i] is used in Y[n, j] for every j, so **X̄ = Ȳ Wᵀ**, with shape (N, D_in). ✓

Shape check as a memory aid: there is exactly one way to combine Ȳ with X or W
(transposed as needed) that produces the right shape. If your shapes work out,
the formula is almost always right.

Why the batch dimension is summed in W̄: the loss is a mean over samples, so
its gradient is the mean of the per-sample gradients. The 1/N is applied once,
in the loss gradient, and every later layer just sums.

**ReLU**: Ȳ ⊙ 1[X > 0], elementwise. No matrix is involved, because the
Jacobian is diagonal.

## 2. Softmax cross-entropy: deriving ∂L/∂z = p − y

**Setup, for one sample.** Logits z ∈ ℝᴷ. True class c, with one-hot y (y_c = 1, all other y_k = 0).

- softmax: pₖ = e^{zₖ} / S, with S = Σⱼ e^{zⱼ}
- loss: ℓ = −log p_c = −z_c + log S

**Derivative**, straight from the second form:

∂ℓ/∂zₖ = −∂z_c/∂zₖ + (1/S) ∂S/∂zₖ = −1[k = c] + e^{zₖ}/S = **pₖ − yₖ**

So **∂ℓ/∂z = p − y**. For the batch mean L = (1/N) Σₙ ℓₙ: **∂L/∂Z = (P − Y)/N**,
of shape (N, K).

**The long way, through the softmax Jacobian** (worth doing once, to see why
everything cancels):

- ∂pᵢ/∂zₖ = pᵢ(1[i = k] − pₖ) (quotient rule). This is a full K×K matrix: every
  output depends on every input.
- ∂ℓ/∂pᵢ = −yᵢ / pᵢ
- ∂ℓ/∂zₖ = Σᵢ (−yᵢ/pᵢ) · pᵢ(1[i = k] − pₖ) = −Σᵢ yᵢ 1[i = k] + pₖ Σᵢ yᵢ = −yₖ + pₖ, because Σᵢ yᵢ = 1.

The K×K Jacobian collapses to a vector subtraction. That is why the two are
always implemented as one fused op: it is cheaper, and it never computes
log pᵢ, which would be −∞ if pᵢ underflows to 0.

**Interpretation.** The gradient is "predicted − target". If the net puts
probability 0.9 on the right class, that logit gets gradient −0.1 (push it up,
gently). Each wrong class gets +pₖ (push it down, in proportion to how much
probability it stole). A perfectly confident correct prediction gives zero
gradient.

**Log-sum-exp trick.** e^{z} overflows float32 once z > 88. For any constant m,

log Σⱼ e^{zⱼ} = m + log Σⱼ e^{zⱼ − m}

which is exact. With m = maxⱼ zⱼ, every exponent is ≤ 0, so nothing overflows,
and the largest term is e⁰ = 1, so the sum is ≥ 1 and its log is safe. The
same shift inside softmax cancels in the ratio. `test_softmax_ce_is_stable_for_huge_logits`
checks logits of ±10⁴.

**Sanity value.** If all logits are equal, p = 1/K and ℓ = log K. For MNIST
that is log 10 = 2.303, so the loss at initialization should be close to it.
Our NumPy nets started higher: **2.90** for the MLP and **5.87** for LeNet.
He init on the *output* layer spreads the logits out (measured std at init:
1.5 for the MLP, 3.8 for LeNet), so the network starts out confidently wrong. Some codebases initialise the last layer near zero for
exactly this reason.

## 3. Convolution as matrix multiplication (im2col) and its backward pass

A conv layer with F filters of size C×K×K computes, at every output pixel,
the dot product of each filter with the C×K×K input patch under it.

**im2col** copies every patch into a row: cols, of shape (N·Ho·Wo, C·K·K). The
filters reshape to W_col (F, C·K·K). Then

out = cols · W_colᵀ + b  →  (N·Ho·Wo, F)  →  reshape to (N, F, Ho, Wo)

This is just a Linear layer, so the backward pass is Linear's (section 1):
W̄_col = Ȳᵀ cols, b̄ = Σ Ȳ, cols̄ = Ȳ W_col.

**col2im** maps cols̄ back to an image gradient. Patches overlap (with stride
1, an interior pixel sits in K² patches), so each pixel's gradient is the
**sum** over all patches containing it. It is the "+=" rule again. Formally,
col2im is the **adjoint** (transpose) of the linear map im2col:
⟨im2col(x), G⟩ = ⟨x, col2im(G)⟩, which `test_col2im_is_adjoint_of_im2col` checks.

Cost: im2col multiplies memory by about K²/stride² (25× for 5×5). That is the
price of turning convolution into one fast matrix multiply.

**Max-pool backward**: max is a switch. Each output's gradient goes entirely
to the argmax position of its window, and every other position gets 0. On
ties, PyTorch picks the first maximum in row-major order, and we match that
(tested with integer inputs, which produce many ties).

**"Convolution" in deep learning is cross-correlation** (the kernel is not
flipped). The PyTorch comparison test would catch a flipped kernel, but
finite differences would not: a flipped kernel is still a self-consistent
function. This is why both kinds of test exist.

## 4. Symmetry breaking

If two hidden units in a layer start with identical incoming weights, they
compute identical outputs, receive identical gradients (same inputs, same
local derivatives, same upstream signal), and take identical update steps.
By induction they stay identical **forever**. A layer of 256 identical units
has the expressive power of one unit.

- **Zeros** is the extreme case. Every hidden activation is 0 (and so is the
  ReLU/tanh output, since relu(0) = tanh(0) = 0), so every weight gradient
  except the output bias is 0. Nothing can ever change.
- **Constant ≠ 0** shows symmetry on its own, without the dead signal: the
  activations are fine, but the units never diverge from each other. There is
  a second, sneakier effect at init. All output rows are identical too, so
  every logit is equal, p is uniform, and the gradient flowing into the last
  hidden layer is Σₖ (pₖ − yₖ) · c = c · (1 − 1) = **exactly 0**. At step 0
  only the output layer learns. We measured ‖dL/dW‖ ≈ 10⁻⁹ in every hidden
  layer, which is roundoff, against 0.1 in the output layer. Once the output
  rows differ, the hidden layers do get gradient, but it is identical for
  every unit. So the network trains as if each layer had **one** unit: 30%
  (ReLU) and 38% (tanh) accuracy, with unit diversity exactly 0 after
  training.
- **Random init** breaks the tie. Each unit starts at a different point, so
  each unit learns a different feature. Biases can safely start at 0, because
  the weights already break symmetry.

Dropout breaks symmetry too: different units are dropped on each step, so
identical units would get different gradients. Random init is still the
standard way.

## 5. Vanishing and exploding gradients

Backprop through L layers multiplies L Jacobians:
∂L/∂h₀ = J₁ᵀ J₂ᵀ ⋯ J_Lᵀ ∂L/∂h_L. If each factor scales a vector by about α,
the gradient scales like α^L:

- α < 1: it **vanishes** (0.5¹⁰ ≈ 10⁻³). The early layers barely learn.
- α > 1: it **explodes** (2¹⁰ ≈ 10³). The updates blow up into NaN.

The same holds for the forward signal. Both directions are controlled by the
*variance* of the weights, which is what initialization sets.

**Variance propagation, forward.** Pre-activation zᵢ = Σⱼ Wᵢⱼ xⱼ, with n_in
terms. W and x are independent and W has zero mean, so

Var(z) = n_in · Var(W) · E[x²]

- For **tanh / linear** units near 0, x ≈ z, so E[x²] ≈ Var(z_prev). Keeping
  Var(z) constant needs **Var(W) = 1/n_in**.
- For **ReLU**, x = max(0, z) with z symmetric around 0, so
  E[x²] = ½ Var(z_prev): half the inputs are zeroed. Keeping Var(z) constant
  needs **Var(W) = 2/n_in**. That is **He init** (He et al. 2015).

**Backward.** The same argument applied to gradients gives Var(W) = 1/n_out
(×2 for ReLU).

**Xavier/Glorot** (2010) compromises between the forward and backward
conditions: **Var(W) = 2/(n_in + n_out)**. With n_in = n_out that equals
1/n_in, which is right for tanh but **half of what ReLU needs**. So with
Xavier on ReLU, the activation variance halves every layer, and the std
falls by √2 ≈ 1.41 per layer, or 2⁵ = 32× per 10 layers. We measured the
logit std at init as 0.046 → 0.0014 → 5.7·10⁻⁵ for depth 10 → 20 → 30, i.e.
33× and then 25× per 10 layers. That matches the prediction.

Does it matter? Not at depth 10: Xavier and He both reach about 96%,
because momentum SGD compensates for a 30× scale error. It does at depth 30:
**71 ± 5% (Xavier) vs 94 ± 4% (He)**. The penalty grows exponentially with
depth, which is why He init (and later normalisation layers and residual
connections) were needed for very deep ReLU networks.

**Why He suits ReLU**, in one sentence: ReLU throws away half the signal's
energy, and the factor 2 in 2/n_in puts it back.

## 6. Dropout

During training, zero each unit with probability p, independently for each
example and each step. Scale the survivors by 1/(1−p) ("inverted dropout")
so that E[output] is unchanged and the test-time network needs no
correction.

Why it regularises (Srivastava et al. 2014):

- **No co-adaptation.** A unit can't rely on any particular other unit being
  present, so it must learn features that are useful on their own.
- **Implicit ensemble.** Every step trains a different random sub-network
  ("thinned network"), and there are 2^units of them sharing weights. At test
  time, the full network with expected activations approximates averaging
  that ensemble.

Cost: noisier gradients, so training is slower, and train accuracy measured
*with* dropout active underestimates the model. That's why our training loop
evaluates the training set in eval mode.

**What we measured (C2).** On 10k CIFAR images over 40 epochs, dropout 0.5
raised val accuracy from 60.7 to 69.4%. It was *behind* for the first ~9
epochs: the regularisation pays off only once the plain model starts to
memorise. The 26-point gap that remains says the binding constraint is data,
not capacity control.

## 7. Optimizers

With g = gradient and η = learning rate:

- **SGD**: θ ← θ − η g. One global step size for every parameter.
- **Momentum** (heavy ball, as in PyTorch): v ← μ v + g, θ ← θ − η v. In a
  direction where g is consistent, v converges to g/(1 − μ), so the effective
  step is η/(1 − μ), i.e. **10× for μ = 0.9**. In directions where g flips
  sign (across a ravine), consecutive terms cancel. So momentum speeds up
  consistent directions and damps oscillation.
- **Adam** (Kingma & Ba 2014): keeps exponential moving averages of g
  (m, the first moment) and g² (v, the second moment), with bias correction
  for their zero initialization. Update: θ ← θ − η m̂ / (√v̂ + ε).
  - Dividing by √v̂ normalises each parameter's step to about η, regardless of
    the gradient's scale. Parameters with small gradients get relatively large
    steps.
  - Consequence: Adam's good learning rates are in **absolute parameter
    units** (~1e-3), whereas SGD's depend on the gradient scale (~1e-2 to
    1e-1 here). Comparing optimizers at one shared lr is meaningless, which is
    why we sweep.

**What we measured (C3).** Best lr: SGD 0.1, momentum 0.03, Adam 1e-3. The
Adam/SGD ratio is exactly the predicted 100×. The momentum/SGD ratio is only
3×, not 10×. The η/(1−μ) argument assumes the gradient direction stays
consistent, which holds at small lr. Near the best lr, plain SGD is already
limited by *stability* (0.3 is worse than 0.1), so it can't reach the
equivalent step size. Also, Adam at lr ≥ 1e-2 fails without producing a NaN:
it collapses to chance with a finite loss. "No NaN" is not "healthy".

## 8. Diagnosing underfitting vs overfitting

| signal | underfitting | overfitting |
|---|---|---|
| train loss | high, still falling or stuck | very low (→ 0) |
| train acc vs val acc | both low, small gap | train ≫ val (large gap) |
| val loss over time | falls with train loss | reaches a minimum, then **rises** |
| fix | bigger model, more steps, higher lr, better init | more data, augmentation, dropout, weight decay, early stopping |

Val loss rising while val accuracy stays flat is the *earliest* overfitting
signal. The model becomes over-confident on the examples it gets wrong
before the decisions themselves change. We saw it in the NumPy MLP: val loss
bottomed out at epoch 8 while val accuracy kept creeping up.

## 9. LeNet (LeCun et al. 1998)

The template for modern CNNs: alternating convolution (local, weight-shared
feature detectors) and subsampling (pooling: small translation invariance,
fewer pixels), followed by fully connected layers. Weight sharing is the key
inductive bias. A 5×5 conv layer has 25·C·F weights whatever the image size,
whereas a dense layer on 28×28 pixels needs 784 weights per unit. The 1998
version used scaled tanh units and trainable average-subsampling. We use
ReLU and max-pool, the modern defaults.

---

## Check yourself

1. Derive ∂L/∂z for softmax cross-entropy without looking, and explain why the
   K×K softmax Jacobian disappears.
2. In Linear's backward, why is b̄ a *sum* over the batch and not a mean, even
   though the loss is a mean?
3. Show that for ReLU with Xavier init, the activation std shrinks by about √2
   per layer. What would happen with tanh and He init (std √(2/n_in))?
4. Why does constant (non-zero) initialization fail, even though the
   activations at init look perfectly healthy?
5. Why can finite-difference gradient checks not detect a flipped convolution
   kernel, while the PyTorch comparison can?
6. Momentum with μ = 0.9 and lr η behaves roughly like plain SGD with what lr,
   and in which directions is that approximation wrong?
7. In C3, plain SGD's best run on seed 0 got 51.3%, but the same lr on two new
   seeds got 42.8% and 44.1%. Why should you expect this, and what does it
   imply about reporting "best of a sweep"?
