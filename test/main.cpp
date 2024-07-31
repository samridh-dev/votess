#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <iostream>
#include <string>
#include <libsycl.hpp>

class CustomReporter : public Catch::StreamingReporterBase {
public:
  using StreamingReporterBase::StreamingReporterBase;

  static std::string getDescription() {
    return "Custom reporter.";
  }

  void testRunStarting(Catch::TestRunInfo const& testRunInfo) override {
    StreamingReporterBase::testRunStarting(testRunInfo);
    std::cout << "[test-run] Starting with name: " << testRunInfo.name << '\n';
  }

  void testCaseStarting(Catch::TestCaseInfo const& testInfo) override {
    std::cout << "[test] " << testInfo.name << '\n';
  }

  void testCaseEnded(Catch::TestCaseStats const& testCaseStats) override {
    // Handle test case end logic and verbosity adjustments
    if (m_config->verbosity() == Catch::Verbosity::Quiet) {
      return;
    }

    std::cout << "[result] " << (testCaseStats.totals.assertions.allPassed() ? 
                                 "PASSED" : "FAILED") << '\n';

  }

};

CATCH_REGISTER_REPORTER("custom", CustomReporter)

int main(int argc, char *args[]) {

  {
    sycl::queue q;
    auto device = q.get_device();
    if (!device.has(sycl::aspect::fp64)) {

      std::cerr << "\e[93mWarning: \e[00m" 
                << "Device does not support double precision (fp64). "
                << "Skipping all device fp64 tests.\n";

    }
  }

  return Catch::Session().run(argc, args);

}
