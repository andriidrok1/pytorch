/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string>

#include "include/libkineto.h"
#include "src/ActivityProfilerProxy.h"
#include "src/ConfigLoader.h"
#include "src/DaemonConfigLoader.h"

namespace KINETO_NAMESPACE {
namespace {

// Stands in for a platform's own daemon config loader and reads no config.
struct NoopDaemonConfigLoader final : IDaemonConfigLoader {
  std::string readBaseConfig() override {
    return "";
  }

  std::string readOnDemandConfig(bool /*activities*/) override {
    return "";
  }

  void setCommunicationFabric(bool /*enabled*/) override {}
};

// With a platform daemon loader factory registered, KINETO_USE_DAEMON must
// still initialize the profiler, and libkineto_init must keep an already
// registered profiler instead of replacing it. Drives the process-global
// libkineto::api(), so it assumes a fresh process (tpx and
// gtest_discover_tests run each case in its own).
TEST(InitTest, UseDaemonInitializesRegisteredProfiler) {
  ::setenv("KINETO_USE_DAEMON", "1", 1);
  ConfigLoader::setDaemonConfigLoaderFactory(
      [] { return std::make_unique<NoopDaemonConfigLoader>(); });

  // Stands in for an earlier libkineto_init without KINETO_USE_DAEMON, which
  // this process cannot run because the variable is read once per process.
  auto& api = libkineto::api();
  api.registerProfiler(std::make_unique<ActivityProfilerProxy>(
      /*cpuOnly=*/true, api.configLoader()));
  const auto* profiler = &api.activityProfiler();

  libkineto_init(/*cpuOnly=*/true, /*logOnError=*/true);
  EXPECT_EQ(&api.activityProfiler(), profiler);
  EXPECT_TRUE(api.isProfilerInitialized());

  libkineto_init(/*cpuOnly=*/true, /*logOnError=*/true);
  EXPECT_EQ(&api.activityProfiler(), profiler);
}

} // namespace
} // namespace KINETO_NAMESPACE
