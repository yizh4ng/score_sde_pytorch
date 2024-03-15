import torch
import tensorflow as tf
import tensorflow_probability as tfp
import os
import logging


def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)


def kl_divergence(p, p_logits, q):
  """Computes the Kullback-Liebler divergence between p and q.

  This function uses p's logits in some places to improve numerical stability.

  Specifically:

  KL(p || q) = sum[ p * log(p / q) ]
    = sum[ p * ( log(p)                - log(q) ) ]
    = sum[ p * ( log_softmax(p_logits) - log(q) ) ]

  Args:
    p: A 2-D floating-point Tensor p_ij, where `i` corresponds to the minibatch
      example and `j` corresponds to the probability of being in class `j`.
    p_logits: A 2-D floating-point Tensor corresponding to logits for `p`.
    q: A 1-D floating-point Tensor, where q_j corresponds to the probability of
      class `j`.

  Returns:
    KL divergence between two distributions. Output dimension is 1D, one entry
    per distribution in `p`.

  Raises:
    ValueError: If any of the inputs aren't floating-point.
    ValueError: If p or p_logits aren't 2D.
    ValueError: If q isn't 1D.
  """
  for tensor in [p, p_logits, q]:
    if not tensor.dtype.is_floating:
      tensor_name = tensor if tf.executing_eagerly() else tensor.name
      raise ValueError('Input %s must be floating type.' % tensor_name)
  p.shape.assert_has_rank(2)
  p_logits.shape.assert_has_rank(2)
  q.shape.assert_has_rank(1)
  return tf.reduce_sum(
    input_tensor=p * (tf.nn.log_softmax(p_logits) - tf.math.log(q)), axis=1)
def classifier_score_from_logits(logits, streaming=False):
  """A helper function for evaluating the classifier score from logits."""
  logits = tf.convert_to_tensor(value=logits)
  logits.shape.assert_has_rank(2)

  # Use maximum precision for best results.
  logits_dtype = logits.dtype
  if logits_dtype != tf.float64:
    logits = tf.cast(logits, tf.float64)

  p = tf.nn.softmax(logits)
  q = tf.reduce_mean(input_tensor=p, axis=0)
  kl = kl_divergence(p, logits, q)
  kl.shape.assert_has_rank(1)
  log_score_ops = (tf.reduce_mean(input_tensor=kl),)
  # log_score_ops contains the score value and possibly the update_op. We
  # apply the same operation on all its elements to make sure their value is
  # consistent.
  final_score_tuple = tuple(tf.exp(value) for value in log_score_ops)
  if logits_dtype != tf.float64:
    final_score_tuple = tuple(
      tf.cast(value, logits_dtype) for value in final_score_tuple)

  if streaming:
    return final_score_tuple
  else:
    return final_score_tuple[0]


def _symmetric_matrix_square_root(mat, eps=1e-10):
  """Compute square root of a symmetric matrix.

  Note that this is different from an elementwise square root. We want to
  compute M' where M' = sqrt(mat) such that M' * M' = mat.

  Also note that this method **only** works for symmetric matrices.

  Args:
    mat: Matrix to take the square root of.
    eps: Small epsilon such that any element less than eps will not be square
      rooted to guard against numerical instability.

  Returns:
    Matrix square root of mat.
  """
  # Unlike numpy, tensorflow's return order is (s, u, v)
  s, u, v = tf.linalg.svd(mat)
  # sqrt is unstable around 0, just use 0 in such case
  si = tf.compat.v1.where(tf.less(s, eps), s, tf.sqrt(s))
  # Note that the v returned by Tensorflow is v = V
  # (when referencing the equation A = U S V^T)
  # This is unlike Numpy which returns v = V^T
  return tf.matmul(tf.matmul(u, tf.linalg.tensor_diag(si)), v, transpose_b=True)
def trace_sqrt_product(sigma, sigma_v):
  """Find the trace of the positive sqrt of product of covariance matrices.

  '_symmetric_matrix_square_root' only works for symmetric matrices, so we
  cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
  ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).

  Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
  We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
  Note the following properties:
  (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
     => eigenvalues(A A B B) = eigenvalues (A B B A)
  (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
     => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
  (iii) forall M: trace(M) = sum(eigenvalues(M))
     => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                   = sum(sqrt(eigenvalues(A B B A)))
                                   = sum(eigenvalues(sqrt(A B B A)))
                                   = trace(sqrt(A B B A))
                                   = trace(sqrt(A sigma_v A))
  A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
  use the _symmetric_matrix_square_root function to find the roots of these
  matrices.

  Args:
    sigma: a square, symmetric, real, positive semi-definite covariance matrix
    sigma_v: same as sigma

  Returns:
    The trace of the positive square root of sigma*sigma_v
  """

  # Note sqrt_sigma is called "A" in the proof above
  sqrt_sigma = _symmetric_matrix_square_root(sigma)

  # This is sqrt(A sigma_v A) above
  sqrt_a_sigmav_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_v, sqrt_sigma))

  return tf.linalg.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))
def frechet_classifier_distance_from_activations(
        activations1, activations2, streaming=False):
  """A helper function evaluating the frechet classifier distance."""
  activations1 = tf.convert_to_tensor(value=activations1)
  activations1.shape.assert_has_rank(2)
  activations2 = tf.convert_to_tensor(value=activations2)
  activations2.shape.assert_has_rank(2)

  activations_dtype = activations1.dtype
  if activations_dtype != tf.float64:
    activations1 = tf.cast(activations1, tf.float64)
    activations2 = tf.cast(activations2, tf.float64)

  # Compute mean and covariance matrices of activations.
  m = (tf.reduce_mean(input_tensor=activations1, axis=0),)
  m_w = (tf.reduce_mean(input_tensor=activations2, axis=0),)
  # Calculate the unbiased covariance matrix of first activations.
  num_examples_real = tf.cast(tf.shape(input=activations1)[0], tf.float64)
  sigma = (num_examples_real / (num_examples_real - 1) *
           tfp.stats.covariance(activations1),)
  # Calculate the unbiased covariance matrix of second activations.
  num_examples_generated = tf.cast(
    tf.shape(input=activations2)[0], tf.float64)
  sigma_w = (num_examples_generated / (num_examples_generated - 1) *
             tfp.stats.covariance(activations2),)
  # m, m_w, sigma, sigma_w are tuples containing one or two elements: the first
  # element will be used to calculate the score value and the second will be
  # used to create the update_op. We apply the same operation on the two
  # elements to make sure their value is consistent.

  def _calculate_fid(m, m_w, sigma, sigma_w):
    """Returns the Frechet distance given the sample mean and covariance."""
    # Find the Tr(sqrt(sigma sigma_w)) component of FID
    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

    # Compute the two components of FID.

    # First the covariance component.
    # Here, note that trace(A + B) = trace(A) + trace(B)
    trace = tf.linalg.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    # Next the distance between means.
    mean = tf.reduce_sum(input_tensor=tf.math.squared_difference(
      m, m_w))  # Equivalent to L2 but more stable.
    fid = trace + mean
    if activations_dtype != tf.float64:
      fid = tf.cast(fid, activations_dtype)
    return fid

  result = tuple(
    _calculate_fid(m_val, m_w_val, sigma_val, sigma_w_val)
    for m_val, m_w_val, sigma_val, sigma_w_val in zip(m, m_w, sigma, sigma_w))
  if streaming:
    return result
  else:
    return result[0]

def kernel_classifier_distance_from_activations(activations1,
                                                        activations2,
                                                        max_block_size=1024,
                                                        dtype=None):
  """Kernel "classifier" distance for evaluating a generative model.

  This methods computes the kernel classifier distance from activations of
  real images and generated images. This can be used independently of the
  kernel_classifier_distance() method, especially in the case of using large
  batches during evaluation where we would like to precompute all of the
  activations before computing the classifier distance, or if we want to
  compute multiple metrics based on the same images. It also returns a rough
  estimate of the standard error of the estimator.

  This technique is described in detail in https://arxiv.org/abs/1801.01401.
  Given two distributions P and Q of activations, this function calculates

      E_{X, X' ~ P}[k(X, X')] + E_{Y, Y' ~ Q}[k(Y, Y')]
        - 2 E_{X ~ P, Y ~ Q}[k(X, Y)]

  where k is the polynomial kernel

      k(x, y) = ( x^T y / dimension + 1 )^3.

  This captures how different the distributions of real and generated images'
  visual features are. Like the Frechet distance (and unlike the Inception
  score), this is a true distance and incorporates information about the
  target images. Unlike the Frechet score, this function computes an
  *unbiased* and asymptotically normal estimator, which makes comparing
  estimates across models much more intuitive.

  The estimator used takes time quadratic in max_block_size. Larger values of
  max_block_size will decrease the variance of the estimator but increase the
  computational cost. This differs slightly from the estimator used by the
  original paper; it is the block estimator of https://arxiv.org/abs/1307.1954.
  The estimate of the standard error will also be more reliable when there are
  more blocks, i.e. when max_block_size is smaller.

  NOTE: the blocking code assumes that real_activations and
  generated_activations are both in random order. If either is sorted in a
  meaningful order, the estimator will behave poorly.

  Args:
    activations1: 2D Tensor containing activations. Shape is
      [batch_size, activation_size].
    activations2: 2D Tensor containing activations. Shape is
      [batch_size, activation_size].
    max_block_size: integer, default 1024. The distance estimator splits samples
      into blocks for computational efficiency. Larger values are more
      computationally expensive but decrease the variance of the distance
      estimate. Having a smaller block size also gives a better estimate of the
      standard error.
    dtype: If not None, coerce activations to this dtype before computations.

  Returns:
   The Kernel Inception Distance. A floating-point scalar of the same type
     as the output of the activations.
   An estimate of the standard error of the distance estimator (a scalar of
     the same type).
  """
  activations1.shape.assert_has_rank(2)
  activations2.shape.assert_has_rank(2)
  activations1.shape[1:2].assert_is_compatible_with(activations2.shape[1:2])

  if dtype is None:
    dtype = activations1.dtype
    assert activations2.dtype == dtype
  else:
    activations1 = tf.cast(activations1, dtype)
    activations2 = tf.cast(activations2, dtype)

  # Figure out how to split the activations into blocks of approximately
  # equal size, with none larger than max_block_size.
  n_r = tf.shape(input=activations1)[0]
  n_g = tf.shape(input=activations2)[0]

  n_bigger = tf.maximum(n_r, n_g)
  n_blocks = tf.cast(tf.math.ceil(n_bigger / max_block_size), dtype=tf.int32)

  v_r = n_r // n_blocks
  v_g = n_g // n_blocks

  n_plusone_r = n_r - v_r * n_blocks
  n_plusone_g = n_g - v_g * n_blocks

  sizes_r = tf.concat([
    tf.fill([n_blocks - n_plusone_r], v_r),
    tf.fill([n_plusone_r], v_r + 1),
  ], 0)
  sizes_g = tf.concat([
    tf.fill([n_blocks - n_plusone_g], v_g),
    tf.fill([n_plusone_g], v_g + 1),
  ], 0)

  zero = tf.zeros([1], dtype=tf.int32)
  inds_r = tf.concat([zero, tf.cumsum(sizes_r)], 0)
  inds_g = tf.concat([zero, tf.cumsum(sizes_g)], 0)

  dim = tf.cast(activations1.shape[1], dtype)

  def compute_kid_block(i):
    """Computes the ith block of the KID estimate."""
    r_s = inds_r[i]
    r_e = inds_r[i + 1]
    r = activations1[r_s:r_e]
    m = tf.cast(r_e - r_s, dtype)

    g_s = inds_g[i]
    g_e = inds_g[i + 1]
    g = activations2[g_s:g_e]
    n = tf.cast(g_e - g_s, dtype)

    k_rr = (tf.matmul(r, r, transpose_b=True) / dim + 1)**3
    k_rg = (tf.matmul(r, g, transpose_b=True) / dim + 1)**3
    k_gg = (tf.matmul(g, g, transpose_b=True) / dim + 1)**3
    return (-2 * tf.reduce_mean(input_tensor=k_rg) +
            (tf.reduce_sum(input_tensor=k_rr) - tf.linalg.trace(k_rr)) /
            (m * (m - 1)) +
            (tf.reduce_sum(input_tensor=k_gg) - tf.linalg.trace(k_gg)) /
            (n * (n - 1)))

  ests = tf.map_fn(
    compute_kid_block, tf.range(n_blocks), dtype=dtype, back_prop=False)

  mn = tf.reduce_mean(input_tensor=ests)

  # tf.nn.moments doesn't use the Bessel correction, which we want here
  n_blocks_ = tf.cast(n_blocks, dtype)
  var = tf.cond(
    pred=tf.less_equal(n_blocks, 1),
    true_fn=lambda: tf.constant(float('nan'), dtype=dtype),
    false_fn=lambda: tf.reduce_sum(input_tensor=tf.square(ests - mn)) / (
            n_blocks_ - 1))

  # return mn, tf.sqrt(var / n_blocks_)
  return mn
