# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT-comple a function that run on the current device, GPU"""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """JIT-comple a function for both a CPU or GPU"""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Creates a function that applies a unary operation to each element in a tensor.

        Args:
        ----
            fn (Callable[[float], float]): Unary function to apply to each element

        Returns:
        -------
            Callable[[Tensor], Tensor]: Function that takes an in puttensor and returns
                a new tensor with the operation applied to each element

        """
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Creates a function that applies a binary operation to each element between tensors.

        Args:
        ----
            fn (Callable[[float, float], float]): Binary function to apply elementwise

        Returns:
        -------
            Callable[[Tensor, Tensor], Tensor]: Function that takes two tensors and returns
                their elementwise combination

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Creates a reduction function that applies a binary function across a tensor dimension.

        Args:
        ----
            fn (Callable[[float, float], float]): Binary function for reduction
            start (float, optional): Initial reduction value. Defaults to 0.0.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: Function that takes tensor and dimension index,
                returns reduced tensor.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs matrix multiplication between two tensors.

        Args:
        ----
            a (Tensor): Left tensor of shape (..., n, m)
            b (Tensor): Right tensor of shape (..., m, p)

        Returns:
        -------
            Tensor: Result tensor of shape (..., n, p)

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            output_pos = index_to_position(out_index, out_strides)
            input_pos = index_to_position(in_index, in_strides)
            out[output_pos] = fn(in_storage[input_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            # Given the ordinal, find its corresponding output index
            to_index(i, out_shape, out_index)
            # Find the corresponding input indices using broadcasting
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            # Find corresponding positions to the indices
            output_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            # Calculate the value of the output with a function applied to the correct input
            out[output_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    if i < size:
        cache[pos] = a[i]

    else:
        cache[pos] = 0.0

    stride = BLOCK_DIM // 2
    while stride > 0:
        if pos < stride and pos + stride < size:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride //= 2

    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Practice parallel reduction sum using CUDA shared memory. This function
    is a wrapper for the _sum_practice kernel.

    Args:
    ----
        a (Tensor): Input tensor of shape [size]

    Returns:
    -------
        TensorData: Partially reduced tensor of shape [size // THREADS_PER_BLOCK + 1]

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        if out_pos >= out_size:
            return

        to_index(out_pos, out_shape, out_index)
        o = index_to_position(out_index, out_strides)

        reduce_size = a_shape[reduce_dim]
        cache[pos] = reduce_value

        for s in range(pos, reduce_size, BLOCK_DIM):
            out_index[reduce_dim] = s
            a_pos = 0
            for dim in range(len(a_shape)):
                a_pos += out_index[dim] * a_strides[dim]
            cache[pos] = fn(cache[pos], a_storage[a_pos])

        cuda.syncthreads()

        stride = BLOCK_DIM // 2
        while stride > 0:
            if pos < stride:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride //= 2

        if pos == 0:
            out[o] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # Initialize storage to be a square of size BLOCK_DIM
    shared_stor_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shared_stor_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Final positions
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    # Check to make sure the current threads don't exceed the  of the square
    # This prevents an indexing error
    if i < size and j < size:
        shared_stor_a[i, j] = a[i * size + j]
        shared_stor_b[i, j] = b[i * size + j]
    # Allow all the threads to catch up
    cuda.syncthreads()
    # Check to make sure the current threads don't exceed the  of the square
    if i < size and j < size:
        dot = 0.0
        # Loop over the inner dimension of both squares
        for k in range(size):
            # Compute the parial dot product
            dot += shared_stor_a[i, k] * shared_stor_b[k, j]
        # Output the dot product to global out variable
        out[i * size + j] = dot


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Practice square matrix multiplication using CUDA shared memory.
    This function is a wrapper for the _mm_practice kernel.

    Args:
    ----
        a (Tensor): First input tensor of shape [size, size]
        b (Tensor): Second input tensor of shape [size, size]

    Returns:
    -------
        TensorData: Output tensor data of multiplied matricies of shape [size, size]

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # Set the batch stride in case batches are used
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    dot = 0.0
    # Move across shared dimension by block dim.
    for block_start in range(0, a_shape[2] + BLOCK_DIM - 1, BLOCK_DIM):
        # Move the information in a and b storage to a_shared to b_shared (shared memory)
        # Check if the current output index (i) and local index (pj) exceeds the input a shape
        # We don't want to access the same index twice
        if i < a_shape[1] and (block_start + pj) < a_shape[2]:
            # Find the index for the current a_idx based on the current batch, i and pj
            a_idx = (
                batch * a_batch_stride
                + i * a_strides[1]
                + (block_start + pj) * a_strides[2]
            )
            # Store the current a array value in the shared storage in its local positions
            a_shared[pi, pj] = a_storage[a_idx]
        else:
            # Set the shared storage position to 0 if the local positions exceed the size of the input array a
            a_shared[pi, pj] = 0.0
        # Check if the current output index (j) and local index (pi) exceeds the input b shape
        if (block_start + pi) < b_shape[1] and j < b_shape[2]:
            # Find the index for the current b_idx based on the current batch, j and pi
            b_idx = (
                batch * b_batch_stride
                + (block_start + pi) * b_strides[1]
                + j * b_strides[2]
            )
            # Store the current b array value in the shared storage in its local positions
            b_shared[pi, pj] = b_storage[b_idx]
        else:
            # Set the shared storage position to 0 if the local positions exceed the size of the input array b
            b_shared[pi, pj] = 0.0

        # All all threads to catch up before calculating the dot product
        cuda.syncthreads()

        # Compute dot product for position c[i, j]
        for k in range(min(BLOCK_DIM, a_shape[2] - block_start)):
            dot += a_shared[pi, k] * b_shared[k, pj]
        # All all threads to catch up before setting the global variable
        cuda.syncthreads()

    # Check if the current output indicies are larger than the output shape
    if i < out_shape[1] and j < out_shape[2]:
        # Calculate the output storage index based on the out_strides
        out_idx = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        # Set the global output variable to the dot product
        out[out_idx] = dot


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
