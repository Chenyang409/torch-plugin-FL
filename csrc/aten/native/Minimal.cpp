#include "Minimal.h"

#include <ATen/native/Resize.h>
#include <third_party/flagos/include/flagos.h>

namespace at::native::flagos {

at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_generic(
      size, allocator, pu1_dks, dtype, memory_format_opt);
}

at::Tensor empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_strided_generic(
      size, stride, allocator, pu1_dks, dtype);
}

at::Tensor as_strided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset) {
  // Convert SymInt to Int and use native implementation
  auto int_size = C10_AS_INTARRAYREF_SLOW(size);
  auto int_stride = C10_AS_INTARRAYREF_SLOW(stride);
  std::optional<int64_t> int_offset = storage_offset.has_value()
      ? std::optional<int64_t>(storage_offset->expect_int())
      : std::nullopt;
  return at::native::as_strided_tensorimpl(self, int_size, int_stride, int_offset);
}

const at::Tensor& resize_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::resize_(
      self, C10_AS_INTARRAYREF_SLOW(size), memory_format);
}

at::Tensor _reshape_alias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  return at::native::_reshape_alias(
      self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
}

at::Tensor _copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  TORCH_CHECK(self.defined(), "Source tensor (self) is not defined.");
  TORCH_CHECK(dst.defined(), "Destination tensor (dst) is not defined.");

  // Get contiguous tensors for copy
  at::Tensor self_contig = self.contiguous();
  at::Tensor dst_contig = dst.is_contiguous() ? dst : dst.contiguous();

  size_t nbytes = self_contig.numel() * self_contig.element_size();

  if (self.is_privateuseone() && dst.is_privateuseone()) {
    // Device to device copy (flagos to flagos)
    foMemcpy(dst_contig.data_ptr(), self_contig.data_ptr(), nbytes, foMemcpyDeviceToDevice);
  } else if (self.is_cpu() && dst.is_privateuseone()) {
    // Host to device copy
    foMemcpy(dst_contig.data_ptr(), self_contig.data_ptr(), nbytes, foMemcpyHostToDevice);
  } else if (self.is_privateuseone() && dst.is_cpu()) {
    // Device to host copy
    foMemcpy(dst_contig.data_ptr(), self_contig.data_ptr(), nbytes, foMemcpyDeviceToHost);
  } else if (self.is_privateuseone() && dst.is_cuda()) {
    // flagos to CUDA copy (same GPU memory, device-to-device)
    foMemcpy(dst_contig.data_ptr(), self_contig.data_ptr(), nbytes, foMemcpyDeviceToDevice);
  } else if (self.is_cuda() && dst.is_privateuseone()) {
    // CUDA to flagos copy (same GPU memory, device-to-device)
    foMemcpy(dst_contig.data_ptr(), self_contig.data_ptr(), nbytes, foMemcpyDeviceToDevice);
  } else {
    TORCH_CHECK(false, "Unsupported device combination for copy: ", self.device(), " -> ", dst.device());
  }

  // If dst was not contiguous, copy back
  if (!dst.is_contiguous()) {
    dst.copy_(dst_contig);
  }

  return dst;
}

at::Tensor _copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  at::native::resize_(dst, self.sizes(), std::nullopt);
  return at::native::flagos::_copy_from(self, dst, false);
}

at::Scalar _local_scalar_dense(const at::Tensor& self) {
  // Copy single element from device to host
  TORCH_CHECK(self.numel() == 1, "_local_scalar_dense expects a tensor with 1 element");

  at::Tensor cpu_tensor = at::empty({1}, self.options().device(at::kCPU));
  foMemcpy(cpu_tensor.data_ptr(), self.data_ptr(), self.element_size(), foMemcpyDeviceToHost);

  return cpu_tensor.item();
}

at::Tensor& set_source_Tensor_(at::Tensor& self, const at::Tensor& source) {
  return at::native::set_tensor_(self, source);
}

at::Tensor& set_source_Storage_(at::Tensor& self, at::Storage source) {
  // Directly manipulate TensorImpl to avoid infinite recursion.
  // at::native::set_ internally calls self.set_(source, 0, ...) which
  // dispatches back to set_.source_Storage_storage_offset, and that
  // implementation used to call result.set_(storage) which dispatches
  // back here, causing infinite recursion and stack overflow.
  int64_t new_size =
      static_cast<int64_t>(source.nbytes() / self.dtype().itemsize());
  int64_t stride_val = 1;
  self.unsafeGetTensorImpl()->set_storage_keep_dtype(std::move(source));
  self.unsafeGetTensorImpl()->set_storage_offset(0);
  self.unsafeGetTensorImpl()->set_sizes_and_strides(
      c10::IntArrayRef(&new_size, 1), c10::IntArrayRef(&stride_val, 1));
  return self;
}

at::Tensor& set_source_Storage_storage_offset_(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  // Directly manipulate TensorImpl to avoid infinite recursion.
  // Previously called result.set_(storage) which dispatches through the
  // dispatcher back to set_.source_Storage, creating an infinite loop.
  result.unsafeGetTensorImpl()->set_storage_keep_dtype(std::move(storage));
  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  result.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return result;
}

at::Tensor view(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

at::Tensor contiguous(
    const at::Tensor& self,
    c10::MemoryFormat memory_format) {
  // If already contiguous, return self
  if (self.is_contiguous(memory_format)) {
    return self;
  }

  // Create a new contiguous tensor and copy data
  auto result = at::empty(self.sizes(), self.options().memory_format(memory_format));

  // For flagos tensors, we need to copy element by element through CPU
  // because the source is non-contiguous
  if (self.is_privateuseone()) {
    // Copy to CPU first (handles non-contiguous layout)
    auto cpu_tensor = at::empty(self.sizes(), self.options().device(at::kCPU));

    // Copy non-contiguous data to contiguous CPU tensor
    // Use native copy which handles strided tensors
    auto self_cpu = at::empty(self.sizes(), self.options().device(at::kCPU));

    // Copy each element - for now, use a simple approach: copy via CPU
    // Create contiguous CPU version
    int64_t numel = self.numel();
    if (numel > 0) {
      // Get contiguous view on CPU by iterating
      auto src_data = self.data_ptr();
      auto dst_data = self_cpu.data_ptr();

      // For strided copy, we need to handle it properly
      // Simple approach: copy the underlying storage and reshape
      size_t storage_size = self.storage().nbytes();
      at::Tensor storage_cpu = at::empty({static_cast<int64_t>(storage_size)}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
      foMemcpy(storage_cpu.data_ptr(), self.storage().data(), storage_size, foMemcpyDeviceToHost);

      // Create a CPU tensor with same storage layout as self
      at::Tensor cpu_view = at::empty({0}, self.options().device(at::kCPU));
      cpu_view.set_(
          storage_cpu.storage(),
          self.storage_offset() * self.element_size(),
          self.sizes(),
          self.strides());
      // Reinterpret as the correct dtype
      cpu_view = cpu_view.view(self.scalar_type());

      // Now make it contiguous on CPU
      auto cpu_contig = at::empty(self.sizes(), self.options().device(at::kCPU).memory_format(memory_format));

      // Manual contiguous copy
      // This is slow but correct for non-contiguous tensors
      if (self.dim() == 2) {
        // Optimized 2D case
        int64_t rows = self.size(0);
        int64_t cols = self.size(1);
        int64_t src_stride0 = self.stride(0);
        int64_t src_stride1 = self.stride(1);
        size_t elem_size = self.element_size();

        char* src_base = static_cast<char*>(storage_cpu.data_ptr()) + self.storage_offset() * elem_size;
        char* dst_base = static_cast<char*>(cpu_contig.data_ptr());

        for (int64_t i = 0; i < rows; i++) {
          for (int64_t j = 0; j < cols; j++) {
            char* src = src_base + (i * src_stride0 + j * src_stride1) * elem_size;
            char* dst = dst_base + (i * cols + j) * elem_size;
            memcpy(dst, src, elem_size);
          }
        }
      } else {
        // General case - use PyTorch's copy
        // First get contiguous on CPU
        cpu_contig.copy_(cpu_view);
      }

      // Copy contiguous CPU tensor to flagos device
      size_t nbytes = cpu_contig.numel() * cpu_contig.element_size();
      foMemcpy(result.data_ptr(), cpu_contig.data_ptr(), nbytes, foMemcpyHostToDevice);
    }

    return result;
  }

  // For non-flagos tensors, use native implementation
  result.copy_(self);
  return result;
}

at::Tensor clone(
    const at::Tensor& self,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  auto memory_format = memory_format_opt.value_or(c10::MemoryFormat::Preserve);

  if (memory_format == c10::MemoryFormat::Preserve) {
    if (self.is_contiguous()) {
      auto result = at::empty_like(self);
      size_t nbytes = self.numel() * self.element_size();
      if (nbytes > 0 && self.is_privateuseone()) {
        foMemcpy(result.data_ptr(), self.data_ptr(), nbytes, foMemcpyDeviceToDevice);
      } else if (nbytes > 0) {
        result.copy_(self);
      }
      return result;
    }
    // For non-contiguous with Preserve, fall through to create contiguous copy
    memory_format = c10::MemoryFormat::Contiguous;
  }

  // Create a contiguous result tensor and copy data into it.
  // NOTE: Do NOT call contiguous() here to avoid infinite recursion when
  // contiguous is unregistered from PrivateUse1 (CompositeImplicitAutograd
  // contiguous calls clone, which would call contiguous again).
  auto result = at::empty(self.sizes(), self.options().memory_format(memory_format));
  result.copy_(self);
  return result;
}

at::Tensor _to_copy(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    bool non_blocking,
    std::optional<c10::MemoryFormat> memory_format_opt) {

  TORCH_CHECK(
      !layout_opt.has_value() || self.layout() == layout_opt.value(),
      "to(options) doesn't support converting to a different layout, "
      "but got self.layout being ",
      self.layout(),
      " and options.layout set as ",
      layout_opt.value());
  TORCH_CHECK(
      self.layout() == c10::kStrided,
      "flagos _to_copy only supports strided tensors, but got ",
      self.layout());
  TORCH_CHECK(
      !self.is_quantized(),
      "flagos _to_copy does not support quantized tensors yet");
  TORCH_CHECK(
      !pin_memory_opt.value_or(false),
      "flagos _to_copy does not support pin_memory=True yet");

  // Determine target device and dtype.
  auto device = device_opt.value_or(self.device());
  auto dtype = dtype_opt.value_or(self.scalar_type());
  auto memory_format = memory_format_opt.value_or(c10::MemoryFormat::Preserve);

  if ((device.is_privateuseone() || device.is_cuda()) && device.index() < 0) {
    const auto self_device = self.device();
    const auto device_index = self_device.type() == device.type() &&
            self_device.index() >= 0
        ? self_device.index()
        : 0;
    device = c10::Device(device.type(), device_index);
  }

  // If no device change and no dtype change, just return contiguous copy
  if (device == self.device() && dtype == self.scalar_type()) {
    if (memory_format == c10::MemoryFormat::Preserve) {
      return self.clone();
    }
    return self.clone().contiguous(memory_format);
  }

  // Handle cross-device copies involving flagos
  bool src_is_flagos = self.is_privateuseone();
  bool dst_is_flagos = device.is_privateuseone();
  bool dst_is_cuda = device.is_cuda();
  bool dst_is_cpu = device.is_cpu();

  at::Tensor result;

  if (src_is_flagos && dst_is_cuda) {
    // flagos -> CUDA: same GPU memory, create CUDA tensor and copy
    int device_index = device.index() >= 0 ? device.index() : (self.device().index() >= 0 ? self.device().index() : 0);
    // First copy with original dtype, then convert if needed
    at::Tensor self_contig = self.contiguous();
    at::Tensor temp = at::empty(self_contig.sizes(), self_contig.options().device(c10::Device(c10::kCUDA, device_index)));
    size_t nbytes = self_contig.numel() * self_contig.element_size();
    if (nbytes > 0) {
      foMemcpy(temp.data_ptr(), self_contig.data_ptr(), nbytes, foMemcpyDeviceToDevice);
    }
    // Handle dtype conversion on CUDA if needed
    if (dtype != self.scalar_type()) {
      result = temp.to(dtype);
    } else {
      result = temp;
    }
  } else if (src_is_flagos && dst_is_flagos) {
    // flagos -> flagos (possibly different device index or dtype)
    int device_index = device.index() >= 0 ? device.index() : 0;
    at::Tensor self_contig = self.contiguous();

    if (dtype != self.scalar_type()) {
      // Need dtype conversion: copy to CPU, convert, copy back
      size_t nbytes = self_contig.numel() * self_contig.element_size();
      at::Tensor cpu_tensor = at::empty(self_contig.sizes(), self_contig.options().device(at::kCPU));
      if (nbytes > 0) {
        foMemcpy(cpu_tensor.data_ptr(), self_contig.data_ptr(), nbytes, foMemcpyDeviceToHost);
      }
      cpu_tensor = cpu_tensor.to(dtype);
      result = at::empty(cpu_tensor.sizes(), cpu_tensor.options().device(c10::Device(c10::kPrivateUse1, device_index)));
      size_t result_nbytes = cpu_tensor.numel() * cpu_tensor.element_size();
      if (result_nbytes > 0) {
        foMemcpy(result.data_ptr(), cpu_tensor.data_ptr(), result_nbytes, foMemcpyHostToDevice);
      }
    } else {
      // Same dtype, just copy
      result = at::empty(self_contig.sizes(), self_contig.options().device(c10::Device(c10::kPrivateUse1, device_index)));
      size_t nbytes = self_contig.numel() * self_contig.element_size();
      if (nbytes > 0) {
        foMemcpy(result.data_ptr(), self_contig.data_ptr(), nbytes, foMemcpyDeviceToDevice);
      }
    }
  } else if (src_is_flagos && dst_is_cpu) {
    // flagos -> CPU
    at::Tensor self_contig = self.contiguous();
    // First copy with original dtype
    at::Tensor temp = at::empty(self_contig.sizes(), self_contig.options().device(at::kCPU));
    size_t nbytes = self_contig.numel() * self_contig.element_size();
    if (nbytes > 0) {
      foMemcpy(temp.data_ptr(), self_contig.data_ptr(), nbytes, foMemcpyDeviceToHost);
    }
    // Then convert dtype if needed
    if (dtype != self.scalar_type()) {
      result = temp.to(dtype);
    } else {
      result = temp;
    }
  } else if (!src_is_flagos && dst_is_flagos) {
    // CPU/CUDA -> flagos
    int device_index = device.index() >= 0 ? device.index() : 0;
    at::Tensor src_contig = self.contiguous();

    // Convert dtype on source device first if needed
    if (dtype != self.scalar_type()) {
      src_contig = src_contig.to(dtype);
    }

    result = at::empty(src_contig.sizes(), src_contig.options().device(c10::Device(c10::kPrivateUse1, device_index)));
    size_t nbytes = src_contig.numel() * src_contig.element_size();
    if (nbytes > 0) {
      if (self.is_cpu()) {
        foMemcpy(result.data_ptr(), src_contig.data_ptr(), nbytes, foMemcpyHostToDevice);
      } else if (self.is_cuda()) {
        foMemcpy(result.data_ptr(), src_contig.data_ptr(), nbytes, foMemcpyDeviceToDevice);
      } else {
        TORCH_CHECK(false, "_to_copy: unsupported source device ", self.device());
      }
    }
  } else {
    // Other combinations: fall back to CPU path
    at::Tensor cpu_tensor = self.to(at::kCPU).to(dtype);
    if (dst_is_flagos) {
      int device_index = device.index() >= 0 ? device.index() : 0;
      result = at::empty(cpu_tensor.sizes(), cpu_tensor.options().device(c10::Device(c10::kPrivateUse1, device_index)));
      size_t nbytes = cpu_tensor.numel() * cpu_tensor.element_size();
      if (nbytes > 0) {
        foMemcpy(result.data_ptr(), cpu_tensor.data_ptr(), nbytes, foMemcpyHostToDevice);
      }
    } else {
      result = cpu_tensor.to(device);
    }
  }

  // Apply memory format if needed
  if (memory_format != c10::MemoryFormat::Preserve) {
    result = result.contiguous(memory_format);
  }

  return result;
}

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // For now, delegate all operations to CPU fallback
  // FlagGems operators will be registered separately in Python
  at::native::cpu_fallback(op, stack);
}

} // namespace at::native::flagos
