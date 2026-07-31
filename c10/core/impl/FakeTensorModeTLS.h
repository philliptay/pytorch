#pragma once
#include <c10/core/TensorImpl.h>
#include <c10/macros/Export.h>

namespace c10::impl {

class C10_API FakeTensorModeTLS {
 public:
  static void set_state(std::shared_ptr<FakeTensorMode> state);
  static void create_state(std::shared_ptr<FakeTensorMode> state);
  static std::shared_ptr<FakeTensorMode> get_state();
  static void reset_state();
};

// [in_kernel_invocation]
// Mirrors Python FakeTensorMode.in_kernel_invocation: while a meta kernel runs
// on a fake tensor's meta backing, device queries must report the meta device
// so the kernel checks devices and allocates outputs on meta. Outside that
// window a fake tensor reports its fake device, so user code like `if
// x.is_cuda` behaves as if the tensor were real.
//
// This is deliberately not keyed off the Fake dispatch key being excluded:
// excluding Fake only suspends the fake layer, which is also what no_dispatch()
// does, and under no_dispatch() a fake tensor must still report its fake
// device.
C10_API bool fake_in_kernel_invocation();
C10_API void set_fake_in_kernel_invocation(bool value);

struct C10_API FakeInKernelInvocationGuard {
  FakeInKernelInvocationGuard() : prev_(fake_in_kernel_invocation()) {
    set_fake_in_kernel_invocation(true);
  }
  ~FakeInKernelInvocationGuard() {
    set_fake_in_kernel_invocation(prev_);
  }
  FakeInKernelInvocationGuard(const FakeInKernelInvocationGuard&) = delete;
  FakeInKernelInvocationGuard& operator=(const FakeInKernelInvocationGuard&) =
      delete;
  FakeInKernelInvocationGuard(FakeInKernelInvocationGuard&&) = delete;
  FakeInKernelInvocationGuard& operator=(FakeInKernelInvocationGuard&&) =
      delete;

 private:
  bool prev_;
};

} // namespace c10::impl
